import os

from typing import Tuple

from d3l_extension import QueryEngine
from tqdm import tqdm

from tabert_cl_training.train_model import TableEmbeddingModule
from util_common.data_loader import CSVDataLoader
from util_common.logging import custom_logger, log_query_and_ground_truth
from util_common.query import aggregate_func, eval_search_results


def load_pylon_tabert_model(ckpt_path: str):
    return TableEmbeddingModule.load_from_checkpoint(ckpt_path).model


def topk_search_and_eval(query_engine: QueryEngine, dataloader: CSVDataLoader, output_dir: str, k: int) -> Tuple[float]:
    queries = dataloader.get_queries()
    ground_truth = dataloader.get_ground_truth()

    num_queries = 0
    precision, recall = [], []

    for qt_name in tqdm(queries):
        query_table = dataloader.read_table(table_name=qt_name)
        results = query_engine.table_query(table=query_table, aggregator=aggregate_func, k=k, verbose=False)

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
    
    return num_queries, avg_precision, avg_recall