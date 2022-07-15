<?php

namespace OCA\Recognize\Classifiers;

use OCA\Recognize\Service\Logger;
use OCP\IConfig;
use Psr\Log\LoggerInterface;
use Symfony\Component\Process\Exception\ProcessTimedOutException;
use Symfony\Component\Process\Exception\RuntimeException;
use Symfony\Component\Process\Process;

class Classifier {
	private LoggerInterface $logger;
	private IConfig $config;

	public function __construct(Logger $logger, IConfig $config) {
		$this->logger = $logger;
		$this->config = $config;
	}

	/**
	 * @param string $model
	 * @param string[] $paths
	 * @param int $timeout
	 * @return \Generator
	 */
	public function classifyFiles(string $model, array $paths, int $timeout): \Generator {
		$this->logger->debug('Classifying '.var_export($paths, true));

		$command = [
			$this->config->getAppValue('recognize', 'node_binary'),
			dirname(__DIR__, 2) . '/src/classifier_'.$model.'.js',
			'-'
		];

		$this->logger->debug('Running '.var_export($command, true));

		$proc = new Process($command, __DIR__);
		if ($this->config->getAppValue('recognize', 'tensorflow.gpu', 'false') === 'true') {
			$proc->setEnv(['RECOGNIZE_GPU' => 'true']);
		}
		if ($this->config->getAppValue('recognize', 'tensorflow.purejs', 'false') === 'true') {
			$proc->setEnv(['RECOGNIZE_PUREJS' => 'true']);
		}
		$proc->setTimeout($timeout);
		$proc->setInput(implode("\n", $paths));
		try {
			$proc->start();

			// Set cores
			$cores = $this->config->getAppValue('recognize', 'tensorflow.cores', '0');
			if ($cores !== '0') {
				@exec('taskset -cp ' . implode(',', range(0, (int)$cores, 1)) . ' ' . $proc->getPid());
			}

			$i = 0;
			$errOut = '';
			$buffer = '';
			foreach ($proc as $type => $data) {
				if ($type !== $proc::OUT) {
					$errOut .= $data;
					$this->logger->debug('Classifier process output: '.$data);
					continue;
				}
				$buffer .= $data;
				$lines = explode("\n", $buffer);
				$buffer = '';
				foreach ($lines as $result) {
					if (trim($result) === '') {
						continue;
					}
					try {
						json_decode($result, true, 512, JSON_OBJECT_AS_ARRAY | JSON_THROW_ON_ERROR | JSON_INVALID_UTF8_IGNORE);
						$invalid = false;
					} catch (\JsonException $e) {
						$invalid = true;
					}
					if ($invalid) {
						$buffer .= "\n".$result;
						continue;
					}
					$this->logger->debug('Result for ' . basename($paths[$i]) . ' = ' . $result);
					try {
						// decode json
						$results = json_decode($result, true, 512, JSON_OBJECT_AS_ARRAY | JSON_THROW_ON_ERROR | JSON_INVALID_UTF8_IGNORE);
						yield $i => $results;
					} catch (\JsonException $e) {
						$this->logger->warning('JSON exception');
						$this->logger->warning($e->getMessage());
						$this->logger->warning($result);
					}
					$i++;
				}
			}
			if ($i !== count($paths)) {
				$this->logger->warning('Classifier process output: '.$errOut);
				throw new \RuntimeException('Classifier process error');
			}
		} catch (ProcessTimedOutException $e) {
			$this->logger->warning($proc->getErrorOutput());
			throw new \RuntimeException('Classifier process timeout');
		} catch (RuntimeException $e) {
			$this->logger->warning($proc->getErrorOutput());
			throw new \RuntimeException('Classifier process could not be started');
		}
	}
}
