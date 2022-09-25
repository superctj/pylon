import argparse
import logging
import os
import sys

from d3l.indexing.similarity_indexes import ClrEmbeddingIndex
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add project root directory to dependency search paths
from util_common.logging import custom_logger
from util_common.topk_query import aggregate_func


def main(args):
    # CSV data loader
    dataloader = CSVDataLoader(
        root_path=args.source_dir,
        sep=",", 
        lineterminator="\n" # this is crucial see https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
    )

    model_name = args.model_path.split("/")[-1][:-5]
    clr_embedding_index_name = f"{model_name}_sample_{args.num_samples}_lsh_{args.lsh_threshold}"
    clr_embedding_index_path = os.path.join(args.index_dir, f"{clr_embedding_index_name}.lsh")
    
    if os.path.exists(clr_embedding_index_path):
        clr_embedding_index = unpickle_python_object(clr_embedding_index_path)
        print(f"{clr_embedding_index_name} Embedding Index: LOADED!")
    else:
        print(f"{clr_embedding_index_name} Embedding Index: STARTED!")
        
        clr_embedding_index = ClrEmbeddingIndex(
            embedding_dim=args.embedding_dim,
            ckpt_path=args.model_path,
            num_samples=args.num_samples,
            dataloader=dataloader,
            index_similarity_threshold=args.lsh_threshold,
            index_cache_dir="./")
        pickle_python_object(clr_embedding_index, clr_embedding_index_path)
        
        print(f"{clr_embedding_index_name} Embedding Index: SAVED!")
    
    query_dataloader = CSVDataLoader(
        root_path=args.target_dir,
        sep=",",
        lineterminator="\n"
    )
    qe = QueryEngine(clr_embedding_index)
    # qe = QueryEngine(name_index, clr_embedding_index) # clr embedding index needs to be put at the end due to the ad-hoc way of changing the source code
    
    with open(args.query_file, "r") as f:
        queries = f.readlines()
        queries = [q.rstrip() for q in queries] # without .csv postfix

    output_dir = os.path.join(args.output_dir, f"{clr_embedding_index_name}_topk_{args.top_k}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory already exists!")

    for i, qt_name in enumerate(tqdm(queries)):
        output_file = os.path.join(output_dir, f"q{i+1}.txt")
        logger = custom_logger(output_file, level=logging.INFO)
        logger.info(args) # For reproduction
        
        logger.info(f"Target table: {qt_name}")
        query_table = query_dataloader.read_table(table_name=qt_name)
        results = qe.table_query_with_clr(table=query_table, aggregator=aggregate_func, k=args.top_k, verbose=False)
        # results, extended_results = qe.table_query_with_clr(table=query_table, aggregator=aggregate_func, k=args.top_k, verbose=True)

        for res in results:
            logger.info(f"{res[0]} {str(res[1])}")


if __name__ == "__main__":
    # "paper_publication", "job_posting"
    parser = argparse.ArgumentParser(description="Top-k Table Union Search",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source_dir", type=str, default="", help="")
    parser.add_argument("--target_dir", type=str, default="", help="")
    parser.add_argument("--index_dir", type=str, default="", help="")
    parser.add_argument("--output_dir", type=str, default="", help="")

    parser.add_argument("--query_file", type=str, default="", help="")
    parser.add_argument("--model_path", type=str, default="", help="")
    parser.add_argument("--lsh_threshold", type=float, default=0.7, help="")
    parser.add_argument("--top_k", type=int, default=0, help="")
    parser.add_argument("--num_samples", type=int, default=0, help="Maximum number of rows to sample from each table to construct embeddings.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="")

    main(parser.parse_args())