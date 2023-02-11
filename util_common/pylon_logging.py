import argparse
import logging
import os
import shutil

from typing import List, Tuple


def create_new_directory(path: str, force: bool = False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if force:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise FileNotFoundError


def custom_logger(logger_name: str, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    format_string="[%(levelname)s] %(message)s"
    log_format = logging.Formatter(format_string)
    
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="w")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def log_query_and_ground_truth(logger: logging.Logger, query_table_name: str, ground_truth: List[str]):
    logger.info(f"Query table: {query_table_name}")
    logger.info("-" * 50)

    for table_name in ground_truth:
        logger.info(f"Ground truth table: {table_name}")

    logger.info("=" * 80)


def log_search_results(logger: logging.Logger, candidate_table_name: str, score: float):
    logger.info(f"Candidate table: {candidate_table_name}")
    logger.info(f"Candidate score: {score}")
    logger.info("-" * 50)


def log_extended_search_results(logger: logging.Logger, column_search_results: List[Tuple]):
    # column_search_results: [((<query_column_name>, <related_column_name>), [column_level_scores])]
    for res in column_search_results:
        logger.info(f"Query column: {res[0][0]}")
        logger.info(f"Candidate column: {res[0][1]}")
        logger.info(f"Scores: {res[1]}")
        logger.info("=" * 50)


def log_args_and_metrics(logger: logging.Logger, args: argparse.Namespace, metrics: Tuple[int, float, float, float]):
    logger.info(args)
    logger.info("=" * 50)
    logger.info(f"{args.dataset_name} top-{args.top_k} search")
    logger.info(f"  Number of queries: {metrics[0]}")
    logger.info(f"  Precision: {metrics[1] : .2f}")
    logger.info(f"  Recall: {metrics[2] : .2f}")
    logger.info(f"  Lookup time: {metrics[3] : .2f} s/query")