import logging
import os
import time

from typing import List, Tuple

from d3l.querying.query_engine import QueryEngine
from tqdm import tqdm

from util_common.data_loader import CSVDataLoader
from util_common.pylon_logging import custom_logger, log_query_and_ground_truth, log_search_results


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
    queries = dataloader.get_queries()
    ground_truth = dataloader.get_ground_truth()

    precision, recall = [], []
    num_queries, total_lookup_time = 0, 0

    for qt_name in tqdm(queries):
        query_table = dataloader.read_table(table_name=qt_name)
        # Check if table is empty after preprocessing
        if query_table.empty:
            print(f"Table *{qt_name}* is empty after preprocessing...")
            print("Continue to the next table...")
            print("=" * 80)
            continue

        # LSH lookup and timing
        start_time = time.time()
        results = query_engine.table_query(
            table=query_table, aggregator=aggregate_func, k=k, verbose=False)
        end_time = time.time()
        total_lookup_time += (end_time - start_time)

        # Log query and ground truth
        num_queries += 1
        output_file = os.path.join(output_dir, f"q{num_queries}.txt")
        logger = custom_logger(output_file)

        query_ground_truth = ground_truth[qt_name]["groundtruth"]
        log_query_and_ground_truth(logger, qt_name, query_ground_truth)

        # Evaluate and log top-k search results
        num_corr = eval_search_results(results, query_ground_truth, logger)
        precision.append(num_corr / k)
        recall.append(num_corr / len(query_ground_truth))
    
    avg_precision = sum(precision) / num_queries
    avg_recall = sum(recall) / num_queries
    avg_lookup_time = total_lookup_time / num_queries
    
    return num_queries, avg_precision, avg_recall, avg_lookup_time