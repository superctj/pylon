import logging
import os

from typing import List, Tuple

from d3l.querying.query_engine import QueryEngine

from util_common.data_loader import CSVDataLoader
from util_common.pylon_logging import log_search_results


def aggregate_func(similarity_scores: List[float]) -> float:
    avg_score = sum(similarity_scores) / len(similarity_scores)
    return avg_score


def eval_search_results(results: List[Tuple[str, float]], ground_truth: List[str], logger: logging.Logger) -> int:
    num_corr = 0

    for res in results:
        if res[0] in ground_truth:
            num_corr += 1

        log_search_results(logger, res[0], res[1])
    
    return num_corr


def topk_search_and_eval(query_engine: QueryEngine, dataloader: CSVDataLoader, output_dir: str, k: int) -> Tuple:
    pass