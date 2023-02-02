import argparse
import os
import sys
# Add project root directory to Python dependency search paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# Add WTE training directory to Python dependency search paths
sys.path.append(os.path.join(os.getcwd(), "wte_cl_training"))

from util import create_or_load_index
from util_common.data_loader import SantosLabeledDataLoader
from util_common.pylon_logging import create_new_directory, custom_logger, log_args_and_metrics
from util_common.query import topk_search_and_eval, QueryEngine


def main(args):
    # Create CSV data loader
    dataloader = SantosLabeledDataLoader(
        dataset_dir=args.dataset_dir,
        query_dir=args.query_dir,
        ground_truth_file=args.ground_truth_file
    )

    # Create or load embedding index
    embedding_index, index_name = create_or_load_index(dataloader, args)

    # Create output directory (overwrite the directory if exists)
    output_dir = os.path.join(
        args.output_dir, f"{index_name}_topk_{args.top_k}")
    create_new_directory(output_dir, force=True)

    # Top-k search and evaluation
    query_engine = QueryEngine(embedding_index)
    metrics = topk_search_and_eval(query_engine, dataloader, output_dir, args.top_k)

    # Log command-line arguments for reproduction and metrics
    meta_log_file = os.path.join(output_dir, "log.txt")
    meta_logger = custom_logger(meta_log_file)
    log_args_and_metrics(meta_logger, args, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-k Table Union Search",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input specification
    parser.add_argument("--dataset_name", type=str, default="", help="")
    parser.add_argument("--dataset_dir", type=str, default="", help="")
    parser.add_argument("--query_dir", type=str, default="", help="")
    parser.add_argument("--ground_truth_file", type=str, default="", help="")
    # Output specification
    parser.add_argument("--index_dir", type=str, default="", help="")
    parser.add_argument("--output_dir", type=str, default="", help="")
    # Embedding model specification
    parser.add_argument("--encoder_path", type=str, default="", help="")
    parser.add_argument("--ckpt_path", type=str, default="", help="")
    parser.add_argument("--embedding_dim", type=int, default=128, help="")
    parser.add_argument("--num_samples", type=int, default=-1, help="Maximum number of rows to sample from each table to construct embeddings.")
    # LSH specification
    parser.add_argument("--lsh_threshold", type=float, default=0.7, help="")
    parser.add_argument("--top_k", type=int, default=0, help="")

    main(parser.parse_args())