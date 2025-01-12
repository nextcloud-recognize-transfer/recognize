<?php
/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Command;

use OCA\Recognize\Classifiers\Audio\MusicnnClassifier;
use OCA\Recognize\Classifiers\Classifier;
use OCA\Recognize\Classifiers\Images\ClusteringFaceClassifier;
use OCA\Recognize\Classifiers\Images\ImagenetClassifier;
use OCA\Recognize\Classifiers\Video\MovinetClassifier;
use OCA\Recognize\Db\QueueFile;
use OCA\Recognize\Exception\Exception;
use OCA\Recognize\Service\Logger;
use OCA\Recognize\Service\SettingsService;
use OCA\Recognize\Service\StorageService;
use OCA\Recognize\Service\TagManager;
use OCP\Files\Config\ICachedMountInfo;
use OCP\Files\Config\IUserMountCache;
use OCP\Files\IRootFolder;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Exception\ExceptionInterface;
use Symfony\Component\Console\Input\ArrayInput;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

class Classify extends Command {
	/** @var array<string, Classifier>  */
	private array $classifiers = [];

	protected IRootFolder $rootFolder;

	public function __construct(
		private StorageService $storageService,
		private TagManager $tagManager,
		private Logger $logger,
		IRootFolder $rootFolder,
		ImagenetClassifier $imagenet,
		ClusteringFaceClassifier $faces,
		MovinetClassifier $movinet,
		MusicnnClassifier $musicnn,
		private IUserMountCache $userMountCache,
		private SettingsService $settings,
		private ClearBackgroundJobs $clearBackgroundJobs,
	) {
		parent::__construct();
		$this->rootFolder = $rootFolder;
		$this->classifiers[ImagenetClassifier::MODEL_NAME] = $imagenet;
		$this->classifiers[ClusteringFaceClassifier::MODEL_NAME] = $faces;
		$this->classifiers[MusicnnClassifier::MODEL_NAME] = $musicnn;
		$this->classifiers[MovinetClassifier::MODEL_NAME] = $movinet;
		// Landmarks are currently processed out of band in a background job, because imagenet schedules it directly
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('recognize:classify')
			->setDescription('Classify all files with the current settings in one go (will likely take a long time).')
			->addOption('retry', null, InputOption::VALUE_NONE, "Only classify untagged images.")
			->addOption('user', 'u', InputOption::VALUE_REQUIRED, "Only classify files for the given user.", null)
			->addOption('path', 'p', InputOption::VALUE_REQUIRED, "Only classify files for the given path. Applicable only with the --user option.", null);
	}

	/**
	 * Execute the command
	 *
	 * @param InputInterface $input
	 * @param OutputInterface $output
	 *
	 * @return int
	 * @throws ExceptionInterface
	 */
	protected function execute(InputInterface $input, OutputInterface $output): int {
		$this->logger->setCliOutput($output);

		// pop "retry" flag from parameters passed to clear background jobs
		$clearBackgroundJobs = new ArrayInput([]);
		$this->clearBackgroundJobs->run($clearBackgroundJobs, $output);

		$models = array_values(array_filter([
			ClusteringFaceClassifier::MODEL_NAME,
			ImagenetClassifier::MODEL_NAME,
			MovinetClassifier::MODEL_NAME,
			MusicnnClassifier::MODEL_NAME,
		], fn ($modelName) => $this->settings->getSetting($modelName . '.enabled') === 'true'));

		$processedTag = $this->tagManager->getProcessedTag();

		$userFilter = $input->getOption('user');
		$pathFilter = $input->getOption('path');

		$pathMatcher = null;

		if ($userFilter === null && $pathFilter !== null) {
			$this->logger->warning('Path filter is set, but no user filter is set. Ignoring path filter.');
			unset($pathFilter);
		} else if ($userFilter !== null) {
			$pathMatcher = '/' . $userFilter;

			if ($pathFilter !== null) {
				if (!str_starts_with($pathFilter, '/')) {
					$pathFilter = '/' . $pathFilter;
				}
				$pathMatcher .= '/files' . $pathFilter;
			}
		}

		if ($pathMatcher !== null) {
			$this->logger->info('Matching files for path: ' . $pathMatcher);
		}

		foreach ($this->storageService->getMounts() as $mount) {
			$this->logger->info('Processing storage ' . $mount['storage_id'] . ' with root ID ' . $mount['override_root']);

			// Setup Filesystem for a users that can access this mount
			$mounts = array_values(array_filter($this->userMountCache->getMountsForStorageId($mount['storage_id']), function (ICachedMountInfo $mountInfo) use ($mount) {
				return $mountInfo->getRootId() === $mount['root_id'];
			}));
			if (count($mounts) > 0) {
				\OC_Util::setupFS($mounts[0]->getUser()->getUID());
			}

			$lastFileId = 0;
			do {
				$i = 0;
				$queues = [
					ImagenetClassifier::MODEL_NAME => [],
					ClusteringFaceClassifier::MODEL_NAME => [],
					MovinetClassifier::MODEL_NAME => [],
					MusicnnClassifier::MODEL_NAME => [],
				];
				foreach ($this->storageService->getFilesInMount($mount['storage_id'], $mount['override_root'], $models, $lastFileId) as $file) {

					if ($pathMatcher !== null) {
						$actualFile = $this->rootFolder->getById($file['fileid']);
						if ($actualFile[0] !== null) {
							$path = $actualFile[0]->getPath();

							if (!str_starts_with($path, $pathMatcher)) {
								continue;
							}
							$this->logger->info('Matching file path: ' . $actualFile[0]->getPath());
						} else {
							continue;
						}
					}
					$i++;
					$lastFileId = $file['fileid'];
					$queueFile = new QueueFile();
					$queueFile->setStorageId($mount['storage_id']);
					$queueFile->setRootId($mount['root_id']);
					$queueFile->setFileId($file['fileid']);
					$queueFile->setUpdate(false);

					if ($file['image']) {
						if (in_array(ClusteringFaceClassifier::MODEL_NAME, $models)) {
							$queues[ClusteringFaceClassifier::MODEL_NAME][] = $queueFile;
						}
					}
					// if retry flag is set, skip other classifiers for tagged files
					if ($input->getOption('retry')) {
						$fileTags = $this->tagManager->getTagsForFiles([$lastFileId]);
						// check if processed tag is already in the tags
						if (in_array($processedTag, $fileTags[$lastFileId])) {
							continue;
						}
					}
					if ($file['image']) {
						if (in_array(ImagenetClassifier::MODEL_NAME, $models)) {
							$queues[ImagenetClassifier::MODEL_NAME][] = $queueFile;
						}
					}
					if ($file['video']) {
						if (in_array(MovinetClassifier::MODEL_NAME, $models)) {
							$queues[MovinetClassifier::MODEL_NAME][] = $queueFile;
						}
					}
					if ($file['audio']) {
						if (in_array(MusicnnClassifier::MODEL_NAME, $models)) {
							$queues[MusicnnClassifier::MODEL_NAME][] = $queueFile;
						}
					}
				}

				foreach ($this->classifiers as $modelName => $classifier) {
					try {
						$classifier->setMaxExecutionTime(0);
						$classifier->classify($queues[$modelName]);
					} catch (Exception $e) {
						$this->logger->warning($e->getMessage(), ['exception' => $e]);
					} catch (\RuntimeException $e) {
						$this->logger->info('Temporary error while running ' . $modelName . 'classifier', ['exception' => $e]);
					} catch (\ErrorException $e) {
						$this->logger->info('Error while running ' . $modelName . 'classifier', ['exception' => $e]);
					}
				}
			} while ($i > 0);
			\OC_Util::tearDownFS();
		}
		return 0;
	}
}
