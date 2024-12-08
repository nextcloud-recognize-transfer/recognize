<?php

/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */
declare(strict_types=1);
namespace OCA\Recognize\Service;

use Psr\Log\LoggerInterface;
use Symfony\Component\Console\Output\OutputInterface;

class Logger implements LoggerInterface {
	private LoggerInterface $logger;
	private ?OutputInterface $cliOutput = null;

	public function __construct(LoggerInterface $logger) {
		$this->logger = $logger;
	}

	/**
	 * @param \Symfony\Component\Console\Output\OutputInterface $out
	 * @return $this
	 */
	public function setCliOutput(OutputInterface $out): Logger {
		$this->cliOutput = $out;
		return $this;
	}


	/**
	 * @inheritDoc
	 */
	public function emergency($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->emergency($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function alert($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->alert($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function critical($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->critical($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function error($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->error($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function warning($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->warning($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function notice($message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->notice($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function info($message, array $context = array()) {
		if (isset($this->cliOutput) && !$this->cliOutput->isQuiet()) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->info($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function debug($message, array $context = array()) {
		if (isset($this->cliOutput) && !$this->cliOutput->isQuiet()) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->debug($message, $context);
	}

	/**
	 * @inheritDoc
	 */
	public function log($level, $message, array $context = array()) {
		if (isset($this->cliOutput)) {
			$this->cliOutput->writeln($message);
		}
		$this->logger->log($level, $message, $context);
	}
}
