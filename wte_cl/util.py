import argparse
import os

from d3l.utils.functions import pickle_python_object, unpickle_python_object

from d3l_extension import PylonWteEmbeddingIndex
from util_common.data_loader import CSVDataLoader


def create_or_load_index(dataloader: CSVDataLoader, args: argparse.Namespace) -> PylonWteEmbeddingIndex:
    ckpt_name = args.ckpt_path.split("/")[-1][:-5]
    index_name = f"{ckpt_name}_sample_{args.num_samples}_lsh_{args.lsh_threshold}"
    index_path = os.path.join(args.index_dir, f"{index_name}.lsh")

    if os.path.exists(index_path):
        embedding_index = unpickle_python_object(index_path)
        print(f"{index_name} Embedding Index: LOADED!")
    else:
        print(f"{index_name} Embedding Index: STARTED!")
        
        embedding_index = PylonWteEmbeddingIndex(
            dataloader=dataloader,
            encoder_path=args.encoder_path,
            ckpt_path=args.ckpt_path,
            embedding_dim=args.embedding_dim,
            num_samples=args.num_samples,
            index_similarity_threshold=args.lsh_threshold
        )
        pickle_python_object(embedding_index, index_path)
        print(f"{index_name} Embedding Index: SAVED!")
    
    return embedding_index, index_name