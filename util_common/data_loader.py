import os
import pickle

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd


class CSVDataLoader(ABC):
    @abstractmethod
    def _read_table_names(self) -> List[str]:
        pass

    @abstractmethod
    def _read_queries(self, query_file: str) -> List[str]:
        pass

    @abstractmethod
    def _read_ground_truth(self, ground_truth_file: str) -> Dict:
        pass

    @abstractmethod
    def get_table_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_queries(self) -> List[str]:
        pass

    @abstractmethod
    def get_ground_truth(self) -> List[str]:
        pass

    @abstractmethod
    def read_table(self, table_name: str) -> pd.DataFrame:
        pass


class PylonCSVDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, query_file: str, ground_truth_file: str):
        self.dataset_dir = dataset_dir
        self.table_names = self._read_table_names()
        self.queries = self._read_queries(query_file)
        self.ground_truth = self._read_ground_truth(ground_truth_file)

    def _read_table_names(self) -> List[str]:
        table_names = [
            ".".join(f.split(".")[:-1])
            for f in os.listdir(self.dataset_dir)
            if str(f)[-4:].lower() == ".csv"
        ]
        return table_names

    def _read_queries(self, query_file: str) -> List[str]:
        with open(query_file, "r") as f:
            queries = f.readlines()
            queries = [q.rstrip() for q in queries] # without .csv postfix
        
        return queries
    
    def _read_ground_truth(self, ground_truth_file: str) -> Dict:
        with open(ground_truth_file, "rb") as f:
            ground_truth = pickle.load(f)
        
        return ground_truth
    
    def get_table_names(self) -> List[str]:
        return self.table_names
    
    def get_queries(self) -> List[str]:
        return self.queries
    
    def get_ground_truth(self) -> List[str]:
        return self.ground_truth
    
    def read_table(self, table_name: str, drop_nan: bool = True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, f"{table_name}.csv")
        table = pd.read_csv(
            table_path, delimiter=",", lineterminator="\n", on_bad_lines="skip", **kwargs) # lineterminator="\n" is crucial see https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
        
        # Select only textual columns
        # table = table.select_dtypes(include="object")
        if drop_nan:
            # Drop empty columns
            table.dropna(axis="columns", how="all", inplace=True)
            # Drop rows with any missing value
            table.dropna(axis="index", how="any", inplace=True)
            
            if table.shape[1] < 1:
                print(f"Table has no column left...")
            if table.shape[0] < 3:
                print(f"Table has fewer than 3 rows...")

        return table


class TUSCSVDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, query_file: str, ground_truth_file: str):
        self.dataset_dir = dataset_dir
        self.table_names = self._read_table_names()
        self.queries = self._read_queries(query_file)
        self.ground_truth = self._read_ground_truth(ground_truth_file)

    def _read_table_names(self) -> List[str]:
        table_names = [
            ".".join(f.split(".")[:-1])
            for f in os.listdir(self.dataset_dir)
            if str(f)[-4:].lower() == ".csv"
        ]
        return table_names

    def _read_queries(self, query_file: str) -> List[str]:
        with open(query_file, "r") as f:
            query_recall_pairs = f.readlines()[1:] # skip the header
            queries = [pair.split(",")[0][:-4] for pair in query_recall_pairs] # without .csv postfix

        return queries
    
    def _read_ground_truth(self, ground_truth_file: str) -> Dict:
        with open(ground_truth_file, "rb") as f:
            ground_truth = pickle.load(f)
        
        return ground_truth
    
    def get_table_names(self) -> List[str]:
        return self.table_names
    
    def get_queries(self) -> List[str]:
        return self.queries
    
    def get_ground_truth(self) -> List[str]:
        return self.ground_truth
    
    def read_table(self, table_name: str, drop_nan: bool = True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, f"{table_name}.csv")
        table = pd.read_csv(
            table_path, delimiter=",", lineterminator="\n", on_bad_lines="skip", **kwargs) # lineterminator="\n" is crucial see https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
        
        # Select only textual columns
        # table = table.select_dtypes(include="object")
        if drop_nan:
            # Drop empty columns
            table.dropna(axis="columns", how="all", inplace=True)
            # Drop rows with any missing value
            table.dropna(axis="index", how="any", inplace=True)
            
            if table.shape[1] < 1:
                print(f"Table has no column left...")
            if table.shape[0] < 3:
                print(f"Table has fewer than 3 rows...")

        return table


class SantosLabeledDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, query_dir: str, ground_truth_file: str):
        self.dataset_dir = dataset_dir
        self.table_names = self._read_table_names()
        self.queries = self._read_queries(query_dir)
        self.ground_truth = self._read_ground_truth(ground_truth_file)

    def _read_table_names(self) -> List[str]:
        table_names = [
            ".".join(f.split(".")[:-1])
            for f in os.listdir(self.dataset_dir)
            if str(f)[-4:].lower() == ".csv"
        ]
        return table_names
    
    def _read_queries(self, query_dir: str) -> List[str]:
        queries = [
            ".".join(f.split(".")[:-1])
            for f in os.listdir(query_dir)
            if str(f)[-4:].lower() == ".csv"
        ]
        return queries
    
    def _read_ground_truth(self, ground_truth_file: str) -> Dict:
        gt_file = pd.read_csv(ground_truth_file)

        ground_truth = {}
        for _, row in gt_file.iterrows():
            query_table_name = row["query_table"][:-4]
            gt_table_name = row["data_lake_table"][:-4]

            if query_table_name not in ground_truth:
                ground_truth[query_table_name] = [gt_table_name]
            else:
                ground_truth[query_table_name].append(gt_table_name)
        
        return ground_truth
    
    def get_table_names(self) -> List[str]:
        return self.table_names
    
    def get_queries(self) -> List[str]:
        return self.queries
    
    def get_ground_truth(self) -> List[str]:
        return self.ground_truth
    
    def read_table(self, table_name: str, drop_nan: bool = True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, f"{table_name}.csv")
        table = pd.read_csv(
            table_path, encoding="latin1", on_bad_lines="skip", **kwargs)
        
        # Select only textual columns
        # table = table.select_dtypes(include="object")
        if drop_nan:
            # Drop empty columns
            table.dropna(axis="columns", how="all", inplace=True)
            # Drop rows with any missing value
            table.dropna(axis="index", how="any", inplace=True)
            
            if table.shape[1] < 1:
                print(f"Table has no column left...")
            if table.shape[0] < 3:
                print(f"Table has fewer than 3 rows...")

        return table 


def test_tus_dataloader():
    version = "small"
    data_dir = "/ssd/congtj/data/table-union-search-benchmark/"
    
    dataset_dir = os.path.join(data_dir, f"table_csvs/{version}_benchmark/")
    query_file = os.path.join(data_dir, f"table_csvs/{version}_groundtruth/recall_groundtruth.csv")
    ground_truth_file = os.path.join(data_dir, f"{version}_groundtruth.pkl")

    dataloader = TUSCSVDataLoader(
        dataset_dir=dataset_dir,
        query_file=query_file,
        ground_truth_file=ground_truth_file
    )

    ground_truth = dataloader.get_ground_truth()
    pprint(ground_truth)


def test_santos_labeled_dataloader():
    data_dir = "/ssd/congtj/data/santos_data/benchmark/santos_benchmark/"

    dataset_dir = os.path.join(data_dir, "datalake/")
    query_dir = os.path.join(data_dir, "query/")
    ground_truth_file = "/home/congtj/pylon_metaspace/santos_baseline/groundtruth/LABELED_benchmark_groundtruth.csv"

    dataloader = SantosLabeledDataLoader(
        dataset_dir=dataset_dir,
        query_dir=query_dir,
        ground_truth_file=ground_truth_file
    )

    ground_truth = dataloader.get_ground_truth()
    assert(len(ground_truth["data_mill_a"]) == 14)
    assert(len(ground_truth["job_pay_scales_a"]) == 15)
    pprint(ground_truth)


if __name__ == "__main__":
    from pprint import pprint

    test_tus_dataloader()
    # test_santos_labeled_dataloader()