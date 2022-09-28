import os

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from d3l.indexing.lsh.lsh_index import LSHIndex
from d3l.indexing.similarity_indexes import NameIndex, SimilarityIndex
from d3l.input_output.dataloaders import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from tabert_cl_training.input_prep import InputFormatter
from tabert_cl_training.train_model import TableEmbeddingModule


def load_pylon_tabert_model(ckpt_path: str):
    return TableEmbeddingModule.load_from_checkpoint(ckpt_path).model


class PylonTabertTransformer:
    def __init__(
        self,
        ckpt_path: str,
        embedding_dim: int,
        num_samples: int = None,
        cache_dir: Optional[str] = None
    ):
        self._ckpt_path = ckpt_path
        self._embedding_dimension = embedding_dim
        self._num_samples = num_samples
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(ckpt_path=self._ckpt_path)
        self._input_formatter = self.get_input_formatter(num_samples=self._num_samples)

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model" and k != "_input_formatter"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(ckpt_path=self._ckpt_path)
        self._input_formatter = self.get_input_formatter(num_samples=self._num_samples)

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def get_embedding_model(self, ckpt_path: str):
        model = load_pylon_tabert_model(ckpt_path)
        return model

    def get_input_formatter(self, num_samples: int):
        input_formatter = InputFormatter(BertTokenizer.from_pretrained("bert-base-uncased"), num_rows=num_samples)
        return input_formatter

    def get_embedding_dimension(self):
        return self._embedding_dimension

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        """
        Extract a column embedding for each column in the table

        Parameters
        ----------
        table: pd.DataFrame
            The table to extract column embeddings from.

        Returns
        -------
        np.ndarray
            A Numpy vector representing the mean of all token embeddings.
        """

        tbl_input = self._input_formatter.generate_instance_input(table)
        tbl_tensor_dict = self._input_formatter.collate([tbl_input])

        # Get projected embeddings
        _, embeddings = self._embedding_model.inference(tbl_tensor_dict)
        # print("Embedding shape: ", embeddings.shape)
        # exit()
        embeddings = embeddings[0].detach().cpu().numpy()
        return embeddings


class PylonTabertEmbeddingIndex(SimilarityIndex):
    def __init__(
        self,
        ckpt_path: str,
        dataloader: DataLoader,
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
        super(PylonTabertEmbeddingIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.ckpt_path = ckpt_path
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed
        self.index_cache_dir = index_cache_dir

        self.transformer = PylonTabertTransformer(
            ckpt_path=self.ckpt_path,
            embedding_dim=self.embedding_dim,
            num_samples=self.num_samples,
            cache_dir=self.index_cache_dir
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured data loader.

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
            # print(table_name)
            try:
                table_data = self.dataloader.read_table(table_name)
            except:
                print("=" * 50)
                print(f"Table *{table_name}* cannot be read correctly...")
                print("Continue to the next table...")
                print("=" * 50)
                continue

            column_signatures = self.transformer.transform(table_data)
            column_names = table_data.columns
            # print(column_signatures.shape)
            # print(column_signatures[0].shape)
            # exit()
            for i in range(column_signatures.shape[0]):
                lsh_index.add(input_id=str(table_name) + "!" + str(column_names[i]), input_set=column_signatures[i])

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
        # if is_numeric(query):
        #     return []

        # query_signature = self.transformer.transform(query)
        if len(query) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query, k=k, with_scores=True)


class QueryEngine():
    def __init__(self, *query_backends: SimilarityIndex):
        """
        Create a new querying engine to perform data discovery in datalakes.
        Parameters
        ----------
        query_backends : SimilarityIndex
            A variable number of similarity indexes.
        """

        self.query_backends = query_backends
        self.tabert_embedding_backend = query_backends[-1]

    @staticmethod
    def group_results_by_table(
        target_id: str,
        results: Iterable[Tuple[str, Iterable[float]]],
        table_groups: Optional[Dict] = None,
    ) -> Dict:
        """
        Groups column-based results by table.
        For a given query column, at most one candidate column is considered for each candidate table.
        This candidate column is the one with the highest sum of similarity scores.

        Parameters
        ----------
        target_id : str
            Typically the target column name used to get the results.
        results : Iterable[Tuple[str, Iterable[float]]]
            One or more pairs of column names (including the table names) and backend similarity scores.
        table_groups: Optional[Dict]
            Iteratively created table groups.
            If None, a new dict is created and populated with the current results.

        Returns
        -------
        Dict
            A mapping of table names to similarity scores.
        """

        if table_groups is None:
            table_groups = defaultdict(list)
        candidate_scores = {}
        for result_item, result_scores in results:
            name_components = result_item.split("!") # change "." to "!" as "." often appearsin column name, e.g., abc.def
            table_name, column_name = (
                "!".join(name_components[:-1]),
                name_components[-1:][0],
            )

            candidate_column, existing_scores = candidate_scores.get(
                table_name, (None, None)
            )
            if existing_scores is None or sum(existing_scores) < sum(result_scores):
                candidate_scores[table_name] = (column_name, result_scores)

        for table_name, (candidate_column, result_scores) in candidate_scores.items():
            table_groups[table_name].append(
                ((target_id, candidate_column), result_scores)
            )
        return table_groups

    @staticmethod
    def get_cdf_scores(
        score_distributions: List[np.ndarray], scores: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        score_distributions : List[np.ndarray]
            The samples of all existing scores for each of the LSH backend.
            Each of these has to be sorted in order to extract the
            Empirical Cumulative Distribution Function (ECDF) value.
        scores : np.ndarray
            An array of current scores for which to extract the ECDF values.

        Returns
        -------
        np.ndarray
            A vector of scores of size (1xc).

        """

        def ecdf(samples, values):
            return [
                np.searchsorted(samples[:, j], value, side="right") / len(samples)
                for j, value in enumerate(values)
            ]

        ecdf_weights = []
        for i in range(len(scores)):
            ecdfs = ecdf(score_distributions[i], scores[i])
            ecdf_weights.append(ecdfs)
        ecdf_weights = np.array(ecdf_weights)
        return np.average(scores, axis=0, weights=ecdf_weights)
    
    def column_query(
        self,
        column: pd.Series,
        tabert_embedding: np.ndarray,
        aggregator: Optional[callable] = None,
        k: Optional[int] = None,
    ) -> Iterable[Tuple[str, Iterable[float]]]:
        """
        Perform column-level top-k nearest neighbour search over the configured LSH backends.
        Parameters
        ----------
        column : pd.Series
            The column query as a Pandas Series.
            The series name will give the name queries.
            The series values will give the value queries.
        tabert_embedding : np.ndarray
            Column embedding from TaBERT model
        aggregator: Optional[callable] = None
            An aggregating function used to merge the results of all configured backends.
            If None then all scores are returned.
        k : Optional[int]
            Only the top-k neighbours will be retrieved from each backend.
            Then, these results are aggregated using the aggregator function and the results re-ranked to retrieve
            the top-k aggregated neighbours.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, Iterable[float]]]
            A collection of (column id, aggregated score values) pairs.
            The scores are the values returned by the backends or one aggregated value if an aggregator is passed.
        """

        results = defaultdict(lambda: [0.0] * len(self.query_backends))
        query_name = str(column.name)
        query_value = column.values.tolist()

        for i, backend in enumerate(self.query_backends):
            if isinstance(backend, NameIndex):
                query_results = backend.query(query=query_name, k=k)
            elif isinstance(backend, PylonTabertEmbeddingIndex):
                query_results = backend.query(query=tabert_embedding, k=k)
            else:
                query_results = backend.query(query=query_value, k=k)

            for rid, score in query_results:
                results[rid][i] = score

        if aggregator is None:
            # If not aggregation is used results are sorted by the mean of the scores.
            # Reverse sorting because the scores are similarities.
            results = sorted(
                results.items(),
                key=lambda items: sum(items[1]) / len(self.query_backends),
                reverse=True,
            )
        else:
            results = {rid: [aggregator(scores)] for rid, scores in results.items()}
            # Reverse sorting because the scores are similarities.
            results = sorted(
                results.items(), key=lambda items: items[1][0], reverse=True
            )

        if k is None:
            return results
        return results[:k]

    def table_query(
        self,
        table: pd.DataFrame,
        aggregator: Optional[callable] = None,
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> Union[Iterable[Tuple], Tuple[Iterable[Tuple], Iterable[Tuple]]]:
        """
        Perform table-level top-k nearest neighbour search over the configured LSH backends.
        Note that this functions assumes that the table name is part of the canonical indexed item ids.
        In other words, it considers the first part of the item id separated by a dot to be the table name.
        Parameters
        ----------
        table : pd.DataFrame
            The table query as a Pandas DataFrame.
            Each column will be the subject of a column-based query.
        aggregator: callable
            An aggregating function used to merge the results of all configured backends at table-level.
        k : Optional[int]
            Only the top-k neighbours will be retrieved from each backend.
            Then, these results are aggregated using the aggregator function and the results re-ranked to retrieve
            the top-k aggregated neighbours.
            If this is None all results are retrieved.
        verbose: bool
            Whether or not to also return the detailed scores for each similar column to some query column.

        Returns
        -------
        Union[Iterable[Tuple], Tuple[Iterable[Tuple], Iterable[Tuple]]]
            Pairs of the form (candidate table name, aggregated similarity score).
            If verbosity is required, also return pairs with column-level similarity details.
        """

        tabert_column_embeddings = self.tabert_embedding_backend.transformer.transform(table)

        extended_table_results = None
        score_distributions = {}

        # Table should already be preprocessed.
        for i, column in enumerate(table.columns.to_list()):
            # Column scores are not aggregated when performing table queries.
            if i < tabert_column_embeddings.shape[0]:
                column_results = self.column_query(
                    column=table[column], clr_embedding=tabert_column_embeddings[i], aggregator=None, k=None # None, k
                )

                score_distributions[column] = np.sort(
                    np.array([scores for _, scores in column_results]), axis=0
                )
                extended_table_results = self.group_results_by_table(
                    target_id=column,
                    results=column_results,
                    table_groups=extended_table_results,
                )
            else:
                # Table has too many columns or long cell values, and TaBERT cuts off overflow columns due to the encoding limit
                assert tabert_column_embeddings.shape[0] < len(table.columns.to_list())
                break

        table_results = {}
        for candidate in extended_table_results.keys():
            candidate_scores = np.array(
                [details[1] for details in extended_table_results[candidate]]
            )
            distributions = [
                score_distributions[details[0][0]]
                for details in extended_table_results[candidate]
            ]
            weighted_scores = self.get_cdf_scores(distributions, candidate_scores)
            if aggregator is None:
                table_results[candidate] = weighted_scores.tolist()
            else:
                table_results[candidate] = aggregator(weighted_scores.tolist())

        # Reverse sorting because the scores are similarities.
        table_results = sorted(table_results.items(), key=lambda pair: pair[1], reverse=True)

        if k is not None:
            table_results = table_results[:k]

        if verbose:
            extended_table_results = [(cand, extended_table_results[cand])
                                    for cand, _ in table_results]
            return table_results, extended_table_results
        return table_results