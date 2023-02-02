import os

from typing import Iterable, Optional, Tuple, Union

import numpy as np
# import torch

from d3l.indexing.lsh.lsh_index import LSHIndex
from d3l.indexing.similarity_indexes import SimilarityIndex
from d3l.input_output.dataloaders import DataLoader
from d3l.utils.functions import is_numeric
from tqdm import tqdm

from wte_cl_training.input_prep import InputFormatter
from wte_cl_training.train_model import TableEmbeddingModule


class PylonWteTransformer:
    def __init__(
        self,
        encoder_path: str,
        ckpt_path: str,
        embedding_dim: int,
        num_samples: int,
        cache_dir: Optional[str] = None,
    ):
        self._encoder_path = encoder_path
        self._ckpt_path = ckpt_path
        self._embedding_dim = embedding_dim
        self._num_samples = num_samples
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(self._ckpt_path)
        self._input_formatter = self.get_input_formatter(self._encoder_path, num_samples=self._num_samples)

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model" and k != "_input_formatter"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(self._ckpt_path)
        self._input_formatter = self.get_input_formatter(self._encoder_path, num_samples=self._num_samples)

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def get_embedding_model(self, ckpt_path: str):
        # device = torch.device("cpu")
        # model = TableEmbeddingModule.load_from_checkpoint(ckpt_path).model.to(device)
        model = TableEmbeddingModule.load_from_checkpoint(ckpt_path).model
        return model
    
    def get_input_formatter(self, encoder_path: str, num_samples: int):
        input_formatter = InputFormatter(encoder_path, num_rows=num_samples)
        return input_formatter

    def get_embedding_dimension(self):
        # return self._embedding_model.get_dimension()
        return self._embedding_dim

    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
        Extract a column embedding for each column in the table
        Given that the underlying embedding model is a n-gram based one,
        the number of out-of-vocabulary tokens should be relatively small or zero.
        Parameters
        ----------
        table: pd.DataFrame
            The table to extract column embeddings from.

        Returns
        -------
        np.ndarray
            A Numpy vector representing the mean of all token embeddings.
        """

        """
        It seems from the paper "spaces are replaced by underscores to coalesce the tokens of a cell to a single token" that they encode a cell as a whole and `get_data_vector` does this under the hood
        """

        col_tensor = self._input_formatter.generate_column_input(input_values)
        embeddings = self._embedding_model(col_tensor).detach().cpu().numpy()
        return embeddings


class PylonWteEmbeddingIndex(SimilarityIndex):
    def __init__(
        self,
        dataloader: DataLoader,
        encoder_path: str,
        ckpt_path: str,
        embedding_dim: int,
        num_samples: int,
        index_hash_size: int = 1024,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
        data_root: Optional[str] = None,
        index_cache_dir: Optional[str] = None
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        transformer_token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        transformer_max_df : float
            Percentage of values the token can appear in before it is ignored.
        transformer_stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        transformer_embedding_model_lang : str
            The embedding model language.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        index_cache_dir : str
            A file system path for storing the embedding model.

        """
        super(PylonWteEmbeddingIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.encoder_path = encoder_path
        self.ckpt_path = ckpt_path
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed
        self.index_cache_dir = index_cache_dir

        self.transformer = PylonWteTransformer(
            encoder_path=self.encoder_path,
            ckpt_path=self.ckpt_path,
            embedding_dim=self.embedding_dim,
            num_samples=self.num_samples,
            cache_dir=self.index_cache_dir
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            dimension=self.embedding_dim,
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
        )

        for table_name in tqdm(self.dataloader.get_table_names()):
            # print("="*20)
            # print(table)
            # print("="*20)
            try:
                table_data = self.dataloader.read_table(table_name)

                if table_data.empty:
                    print(f"Table *{table_name}* is empty after preprocessing...")
                    print("Continue to the next table...")
                    print("=" * 80)
                    continue
            except:
                print(f"Table *{table_name}* cannot be read correctly...")
                print("Continue to the next table...")
                print("=" * 50)
                continue

            column_signatures = [
                (c, self.transformer.transform(table_data[c].tolist()))
                for c in table_data.columns
                if not is_numeric(table_data[c]) and table_data[c].count() > 0
            ]

            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table_name) + "!" + str(c), input_set=signature)

        return lsh_index

    def query(
        self, query: np.ndarray, k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )