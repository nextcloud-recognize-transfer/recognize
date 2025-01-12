<?php
/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Service;

use \OCA\Recognize\Vendor\Rubix\ML\Datasets\Labeled;
use \OCA\Recognize\Vendor\Rubix\ML\Kernels\Distance\Euclidean;
use OCA\Recognize\Clustering\HDBSCAN;
use OCA\Recognize\Db\FaceCluster;
use OCA\Recognize\Db\FaceClusterMapper;
use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;

class FaceClusterAnalyzer {
	public const MIN_DATASET_SIZE = 120;
	public const MIN_DETECTION_SIZE = 0.03;
	public const MIN_CLUSTER_SEPARATION = 0.35;
	public const MAX_CLUSTER_EDGE_LENGTH = 0.5;
	public const DIMENSIONS = 1024;
	public const MAX_OVERLAP_NEW_CLUSTER = 0.1;
	public const MIN_OVERLAP_EXISTING_CLUSTER = 0.5;

	private FaceDetectionMapper $faceDetections;
	private FaceClusterMapper $faceClusters;
	private Logger $logger;
	private int $minDatasetSize = self::MIN_DATASET_SIZE;
	private SettingsService $settingsService;

	public function __construct(FaceDetectionMapper $faceDetections, FaceClusterMapper $faceClusters, Logger $logger, SettingsService $settingsService) {
		$this->faceDetections = $faceDetections;
		$this->faceClusters = $faceClusters;
		$this->logger = $logger;
		$this->settingsService = $settingsService;
	}

	public function setMinDatasetSize(int $minSize) : void {
		$this->minDatasetSize = $minSize;
	}

	/**
	 * @throws \OCP\DB\Exception
	 * @throws \JsonException
	 */
	public function calculateClusters(string $userId, int $batchSize = 0): void {
		$this->logger->debug('ClusterDebug: Retrieving face detections for user ' . $userId);

		if ($batchSize === 0) {
			ini_set('memory_limit', '-1');
		}


		$sampledDetections = [];
		$existingClusters = $this->faceClusters->findByUserId($userId);
		/** @var array<int,int> $maxVotesByCluster */
		$maxVotesByCluster = [];
		foreach ($existingClusters as $existingCluster) {
			$sampled = $this->faceDetections->findClusterSample($existingCluster->getId(), $this->getReferenceSampleSize(count($existingClusters)));
			$sampledDetections = array_merge($sampledDetections, $sampled);
			$maxVotesByCluster[$existingCluster->getId()] = count($sampled);
		}

		if ($batchSize > 0) {
			$rejectedDetections = $this->faceDetections->sampleRejectedDetectionsByUserId($userId, $this->getRejectSampleSize($batchSize), self::MIN_DETECTION_SIZE, self::MIN_DETECTION_SIZE);
			$requestedFreshDetectionCount = max($batchSize - count($rejectedDetections) - count($sampledDetections), 500);
			$freshDetections = $this->faceDetections->findUnclusteredByUserId($userId, $requestedFreshDetectionCount, self::MIN_DETECTION_SIZE, self::MIN_DETECTION_SIZE);
		} else {
			$freshDetections = $this->faceDetections->findUnclusteredByUserId($userId, 0, self::MIN_DETECTION_SIZE, self::MIN_DETECTION_SIZE);
			$rejectedDetections = $this->faceDetections->sampleRejectedDetectionsByUserId($userId, $this->getRejectSampleSize(count($freshDetections)), self::MIN_DETECTION_SIZE, self::MIN_DETECTION_SIZE);
		}


		$unclusteredDetections = array_merge($freshDetections, $rejectedDetections);
		$detections = array_merge($unclusteredDetections, $sampledDetections);

		if (count($detections) < $this->minDatasetSize || count($freshDetections) === 0) {
			$this->logger->debug('ClusterDebug: Not enough face detections found');
			return;
		}

		$this->logger->debug('ClusterDebug: Found ' . count($freshDetections) . " fresh detections. Adding " . count($rejectedDetections). " old detections and " . count($sampledDetections). " sampled detections from already existing clusters. Calculating clusters on " . count($detections) . " detections.");


		$dataset = new Labeled(array_map(static function (FaceDetection $detection): array {
			return $detection->getVector();
		}, $detections), array_combine(array_keys($detections), array_keys($detections)), false);

		$dataset->features();

		$n = count($detections);
		$hdbscan = new HDBSCAN($dataset, $this->getMinClusterSize($n), $this->getMinSampleSize($n));

		$numberOfClusteredDetections = 0;
		$clusters = $hdbscan->predict(self::MIN_CLUSTER_SEPARATION, self::MAX_CLUSTER_EDGE_LENGTH);

		foreach ($clusters as $flatCluster) {
			/** @var int[] $detectionKeys */
			$detectionKeys = array_keys($flatCluster->getClusterVertices());

			$clusterDetections = array_filter($detections, function ($key) use ($detectionKeys) {
				return isset($detectionKeys[$key]);
			}, ARRAY_FILTER_USE_KEY);
			$clusterCentroid = self::calculateCentroidOfDetections($clusterDetections);
			$votes = [];

			// Let already clustered detections vote which
			// clusterId these newly clustered detections get
			foreach ($detectionKeys as $detectionKey) {
				if ($detectionKey < count($unclusteredDetections)) {
					continue;
				}

				$vote = $detections[$detectionKey]->getClusterId();

				if ($vote === null) {
					$vote = -1;
				}

				$votes[] = $vote;
			}

			$oldClusterId = -1;
			if (empty($votes)) {
				$overlap = 0.0;
			} else {
				$votes = array_count_values($votes);
				$oldClusterId = array_search(max($votes), $votes);
				$overlap = max($votes) / $maxVotesByCluster[$oldClusterId];
			}

			// If more than X% of already clustered detections are for this, we keep it
			if ($overlap > self::MIN_OVERLAP_EXISTING_CLUSTER) {
				$clusterId = $oldClusterId;
				$cluster = $this->faceClusters->find($clusterId);
			} elseif ($overlap < self::MAX_OVERLAP_NEW_CLUSTER) {
				// otherwise we create a new cluster

				$cluster = new FaceCluster();
				$cluster->setTitle('');
				$cluster->setUserId($userId);
				$this->faceClusters->insert($cluster);
			} else {
				// this is a shit cluster. Don't add to it.
				continue;
			}

			foreach ($detectionKeys as $detectionKey) {
				if ($detectionKey >= count($unclusteredDetections)) {
					// This is a sampled, already clustered detection, ignore.
					continue;
				}

				// If threshold is larger than 0 and $clusterCentroid is not the null vector
				if ($unclusteredDetections[$detectionKey]->getThreshold() > 0.0 && count(array_filter($clusterCentroid, fn ($el) => $el !== 0.0)) > 0) {
					// If a threshold is set for this detection and its vector is farther away from the centroid
					// than the threshold, skip assigning this detection to the cluster
					$distanceValue = self::distance($clusterCentroid, $unclusteredDetections[$detectionKey]->getVector());
					if ($distanceValue >= $unclusteredDetections[$detectionKey]->getThreshold()) {
						continue;
					}
				}

				$this->faceDetections->assocWithCluster($unclusteredDetections[$detectionKey], $cluster);
				$numberOfClusteredDetections += 1;
			}
		}

		$this->logger->debug('ClusterDebug: Clustering complete. Total num of clustered detections: ' . $numberOfClusteredDetections);

		foreach ($unclusteredDetections as $detection) {
			if ($detection->getClusterId() === null) {
				// This detection was run through clustering but wasn't assigned to any cluster
				$detection->setClusterId(-1);
				$this->faceDetections->update($detection);
			}
		}

		$this->settingsService->setSetting('clusterFaces.status', 'true');
		$this->settingsService->setSetting('clusterFaces.lastRun', (string)time());
	}

	/**
	 * @param FaceDetection[] $detections
	 * @return list<float>
	 */
	public static function calculateCentroidOfDetections(array $detections): array {
		// init 1024 dimensional vector
		/** @var list<float> $sum */
		$sum = [];
		for ($i = 0; $i < self::DIMENSIONS; $i++) {
			$sum[] = 0;
		}

		if (count($detections) === 0) {
			return $sum;
		}

		foreach ($detections as $detection) {
			$sum = array_map(static function ($el, $el2) {
				return $el + $el2;
			}, $detection->getVector(), $sum);
		}

		$centroid = array_map(static function ($el) use ($detections) {
			return $el / count($detections);
		}, $sum);

		return $centroid;
	}

	/**
	 * @param array<FaceDetection> $detections
	 * @return array<int,FaceDetection[]>
	 */
	private function findFilesWithDuplicateFaces(array $detections): array {
		$files = [];
		foreach ($detections as $detection) {
			if (!isset($files[$detection->getFileId()])) {
				$files[$detection->getFileId()] = [];
			}
			$files[$detection->getFileId()][] = $detection;
		}

		/** @var array<int,FaceDetection[]> $filesWithDuplicateFaces */
		$filesWithDuplicateFaces = array_filter($files, static function ($detections) {
			return count($detections) > 1;
		});

		return $filesWithDuplicateFaces;
	}

	private static ?Euclidean $distance;

	/**
	 * @param list<int|float> $v1
	 * @param list<int|float> $v2
	 * @return float
	 */
	private static function distance(array $v1, array $v2): float {
		if (!isset(self::$distance)) {
			self::$distance = new Euclidean();
		}
		return self::$distance->compute($v1, $v2);
	}

	/**
	 * Hypothesis is that photos per identity scale with ~ n^(1/5) in the total number of photos
	 * @param int $batchSize
	 * @return int
	 */
	private function getMinClusterSize(int $batchSize) : int {
		return (int)round(max(2, min(5, $batchSize ** (1 / 4.7))));
	}

	/**
	 * We use 4.6 here to have this slightly smaller than MinClusterSize but still scale similarly
	 * @param int $batchSize
	 * @return int
	 */
	private function getMinSampleSize(int $batchSize) : int {
		return (int)round(max(2, min(4, $batchSize ** (1 / 5.6))));
	}

	/**
	 * Grows to ~5000 detections for ~200-800 clusters (detections per cluster drop exponentially)
	 * and then grows linearly with 5 detections per cluster
	 * @param int $numberClusters
	 * @return int
	 */
	private function getReferenceSampleSize(int $numberClusters) : int {
		return (int)round(75 * 2 ** (-0.007 * $numberClusters) + 5);
	}

	private function getRejectSampleSize(int $batchSize): int {
		return (int) min(($batchSize / 4), 12 * $batchSize ** (0.55)); // I love maths. Slap me.
	}
}
