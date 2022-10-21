<?php

namespace OCA\Recognize\BackgroundJobs;

use OCA\Recognize\Service\QueueService;
use OCP\AppFramework\Utility\ITimeFactory;
use OCP\BackgroundJob\IJobList;
use OCP\BackgroundJob\Job;
use OCP\BackgroundJob\TimedJob;
use OCP\DB\Exception;
use OCP\Files\Config\ICachedMountInfo;
use OCP\Files\Config\IUserMountCache;
use OCP\IConfig;
use Psr\Log\LoggerInterface;

abstract class ClassifierJob extends TimedJob {
	private LoggerInterface $logger;
	private QueueService $queue;
	private IUserMountCache $userMountCache;
	private IJobList $jobList;
	/**
	 * @var \OCP\IConfig
	 */
	private IConfig $config;

	public function __construct(ITimeFactory $time, LoggerInterface $logger, QueueService $queue, IUserMountCache $userMountCache, IJobList $jobList, IConfig $config) {
		parent::__construct($time);
		$this->logger = $logger;
		$this->queue = $queue;
		$this->userMountCache = $userMountCache;
		$this->jobList = $jobList;
		$this->config = $config;
		$this->setInterval(60 * 5);
		$this->setTimeSensitivity(self::TIME_INSENSITIVE);
	}

	protected function runClassifier(string $model, $argument) {
		$storageId = $argument['storageId'];
		$rootId = $argument['rootId'];
		if ($this->config->getAppValue('recognize', $model.'.enabled', 'false') !== 'true') {
			$this->logger->debug('Not classifying files of storage '.$storageId. ' using '.$model. ' because model is disabled');
			// `static` to get extending subclass name
			$this->jobList->remove(static::class, $argument);
			return;
		}
		$this->logger->debug('Classifying files of storage '.$storageId. ' using '.$model);
		try {
			$files = $this->queue->getFromQueue($model, $storageId, $rootId, $this->getBatchSize());
		} catch (Exception $e) {
			$this->config->setAppValue('recognize', $model.'.status', 'false');
			$this->logger->error('Cannot retrieve items from '.$model.' queue', ['exception' => $e]);
			return;
		}

		// Setup Filesystem for a users that can access this mount
		$mounts = array_values(array_filter($this->userMountCache->getMountsForStorageId($storageId), function (ICachedMountInfo $mount) use ($rootId) {
			return $mount->getRootId() === $rootId;
		}));
		if (count($mounts) > 0) {
			\OC_Util::setupFS($mounts[0]->getUser()->getUID());
		}

		try {
			$this->classify($files);
		} catch(\Throwable $e) {
			$this->config->setAppValue('recognize', $model.'.status', 'false');
			throw $e;
		}

		try {
			// If there is at least one file left in the queue, reschedule this job
			$files = $this->queue->getFromQueue($model, $storageId, $rootId, 1);
			if (count($files) === 0) {
				// `static` to get extending subclasse name
				$this->jobList->remove(static::class, $argument);
			}
		} catch (Exception $e) {
			$this->config->setAppValue('recognize', $model.'.status', 'false');
			$this->logger->error('Cannot retrieve items from '.$model.' queue', ['exception' => $e]);
			return;
		}
	}

	/**
	 * @return int
	 */
	abstract protected function getBatchSize() : int;

	/**
	 * @param array $files
	 * @return void
	 */
	abstract protected function classify(array $files) : void;
}
