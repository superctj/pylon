import logging
import os
import time

from collections import defaultdict
from typing import Iterable, Tuple, Optional, Dict, Union, List

import numpy as np
import pandas as pd

from d3l.indexing.similarity_indexes import NameIndex, SimilarityIndex
from tqdm import tqdm

from util_common.data_loader import CSVDataLoader
from util_common.pylon_logging import custom_logger, log_query_and_ground_truth, log_search_results, log_extended_search_results


def aggregate_func(similarity_scores: List[float]) -> float:
    avg_score = sum(similarity_scores) / len(similarity_scores)
    return avg_score


class QueryEngine:
    def __init__(self, *query_backends: SimilarityIndex):
        """
        Create a new querying engine to perform data discovery in datalakes.
        Parameters
        ----------
        query_backends : SimilarityIndex
            A variable number of similarity indexes.
        """

        self.query_backends = query_backends

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
            name_components = result_item.split("!")
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

        extended_table_results = None
        score_distributions = {}
        for column in table.columns:
            """Column scores are not aggregated when performing table queries."""
            column_results = self.column_query(
                column=table[column], aggregator=None, k=None # k, None
            )

            score_distributions[column] = np.sort(
                np.array([scores for _, scores in column_results]), axis=0
            )
            extended_table_results = self.group_results_by_table(
                target_id=column,
                results=column_results,
                table_groups=extended_table_results,
            )

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


def eval_search_results(results: List[Tuple[str, float]], ground_truth: List[str], logger: logging.Logger, verbose=False, extended_results=None) -> int:
    num_corr = 0

    for i, res in enumerate(results):
        if res[0] in ground_truth:
            num_corr += 1

        log_search_results(logger, res[0], res[1])
        
        if verbose: # each item in extended_results has the format (<table_name>, [((<query_column_name>, <related_column_name>), [column_level_scores])])
            assert(extended_results[i][0] == res[0])
            log_extended_search_results(logger, extended_results[i][1])

    return num_corr


def topk_search_and_eval(query_engine: QueryEngine, dataloader: CSVDataLoader, output_dir: str, k: int, verbose=False) -> Tuple:
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
        if not verbose:
            results = query_engine.table_query(
                table=query_table, aggregator=aggregate_func, k=k, verbose=verbose)
        else:
            results, extended_results = query_engine.table_query(
                table=query_table, aggregator=aggregate_func, k=k, verbose=verbose)
        end_time = time.time()
        total_lookup_time += (end_time - start_time)

        # Log query and ground truth
        num_queries += 1
        output_file = os.path.join(output_dir, f"q{num_queries}.txt")
        logger = custom_logger(output_file)

        try:
            query_ground_truth = ground_truth[qt_name]["groundtruth"]
        except:
            query_ground_truth = ground_truth[qt_name]
        log_query_and_ground_truth(logger, qt_name, query_ground_truth)

        # Evaluate and log top-k search results
        if not verbose:
            num_corr = eval_search_results(results, query_ground_truth, logger)
        else:
            num_corr = eval_search_results(results, query_ground_truth, logger, verbose, extended_results)
        precision.append(num_corr / k)
        recall.append(num_corr / len(query_ground_truth))
    
    avg_precision = sum(precision) / num_queries
    avg_recall = sum(recall) / num_queries
    avg_lookup_time = total_lookup_time / num_queries
    
    return num_queries, avg_precision, avg_recall, avg_lookup_time