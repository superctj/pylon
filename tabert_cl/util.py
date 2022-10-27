import argparse
import os

from typing import Tuple

from d3l_extension import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from tqdm import tqdm

from d3l_extension import PylonTabertEmbeddingIndex
from util_common.data_loader import CSVDataLoader
from util_common.pylon_logging import custom_logger, log_query_and_ground_truth
from util_common.query import aggregate_func, eval_search_results


def create_or_load_index(dataloader: CSVDataLoader, args: argparse.Namespace) -> PylonTabertEmbeddingIndex:
    ckpt_name = args.ckpt_path.split("/")[-1][:-5]
    index_name = f"{ckpt_name}_sample_{args.num_samples}_lsh_{args.lsh_threshold}"
    index_path = os.path.join(args.index_dir, f"{index_name}.lsh")

    if os.path.exists(index_path):
        embedding_index = unpickle_python_object(index_path)
        print(f"{index_name} Embedding Index: LOADED!")
    else:
        print(f"{index_name} Embedding Index: STARTED!")
        
        embedding_index = PylonTabertEmbeddingIndex(
            ckpt_path=args.ckpt_path,
            dataloader=dataloader,
            embedding_dim=args.embedding_dim,
            num_samples=args.num_samples,
            index_similarity_threshold=args.lsh_threshold
        )
        pickle_python_object(embedding_index, index_path)
        print(f"{index_name} Embedding Index: SAVED!")
    
    return embedding_index, index_name


def topk_search_and_eval(query_engine: QueryEngine, dataloader: CSVDataLoader, output_dir: str, k: int) -> Tuple[float]:
    queries = dataloader.get_queries()
    ground_truth = dataloader.get_ground_truth()

    num_queries = 0
    precision, recall = [], []

    for qt_name in tqdm(queries):
        query_table = dataloader.read_table(table_name=qt_name)
        if query_table.empty:
            print(f"Table *{qt_name}* is empty after preprocessing...")
            print("Continue to the next table...")
            print("=" * 80)
            continue

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