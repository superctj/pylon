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
        table = table.select_dtypes(include="object")
        if drop_nan:
            # Drop empty columns
            table.dropna(axis="columns", how="all", inplace=True)
            # Drop rows with missing values
            table.dropna(axis="index", how="any", inplace=True)
            
            if table.shape[1] < 1:
                print(f"Table has no column left...")
            if table.shape[0] < 3:
                print(f"Table has fewer than 3 rows...")

        return table