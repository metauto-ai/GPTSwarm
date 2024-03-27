#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from loguru import logger
# from globals import CompletionTokens, PromptTokens, Cost
from swarm.utils.const import GPTSWARM_ROOT

def configure_logging(print_level: str = "INFO", logfile_level: str = "DEBUG") -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level (str): The logging level for console output.
        logfile_level (str): The logging level for file output.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(GPTSWARM_ROOT / 'logs/log.txt', level=logfile_level, rotation="10 MB")

def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        mode (str): The mode of operation, used in the file path.
        time_stamp (str): The current timestamp, used in the file path.

    Returns:
        Path: The path to the initialized log file.
    """
    try:
        log_file_path = GPTSWARM_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float,  prompt_tokens: int, complete_tokens: int, log_file_path: str) -> None:
    """
    Custom log function for swarm operations. Includes dynamic global variables.

    Args:
        sender (str): The name of the sender.
        text (str): The text message to log.
        cost (float): The cost associated with the operation.
        result_file (Path, optional): Path to the result file. Default is None.
        solution (list, optional): Solution data to be logged. Default is an empty list.
    """
    # Directly reference global variables for dynamic values
    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    logger.info(formatted_message)

    try:
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise


def main():
    configure_logging()
    # Example usage of swarmlog with dynamic values
    swarmlog("SenderName", "This is a test message.", 0.123)

if __name__ == "__main__":
    main()

