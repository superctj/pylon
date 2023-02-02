import os
import random
random.seed(12345)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from torch.utils.data import Dataset
from util_common.fasttext_web_table_embeddings import FastTextWebTableModel


class InputFormatter(object):
    def __init__(self,
        encoder_path: str,
        num_rows: int = 5,
        ):
        self.num_rows = num_rows
        self.encoder_path = encoder_path
        self.encoder = FastTextWebTableModel.load_model(self.encoder_path)

    def _sample_rows(self, table, num_rows): # Only for training
        if table.shape[0] > num_rows:
            sample = table.sample(n=num_rows)
        else:
            sample = table
        return sample

    def _transform(self, input_values):
        embeddings = [self.encoder.get_data_vector(cell_values) for cell_values in input_values]
        
        if len(embeddings) == 0:
            # return np.empty(0)
            return np.random.randn(self.encoder.get_dimension())
            
        return np.mean(np.array(embeddings), axis=0)

    def generate_input_pairs(self, table, idx):
        t1 = self._sample_rows(table, self.num_rows)
        t2 = self._sample_rows(table, self.num_rows)

        t1_col_embeddings = [
            self._transform(t1[c].tolist())
            for c in t1.columns
            #if not is_numeric(t1[c]) and t1[c].count() > 0
        ]

        t2_col_embeddings = [
            self._transform(t2[c].tolist())
            for c in t2.columns
            #if not is_numeric(t2[c]) and t2[c].count() > 0
        ]

        t1_col_embeddings = np.stack(t1_col_embeddings)
        t2_col_embeddings = np.stack(t2_col_embeddings)

        return t1_col_embeddings, t2_col_embeddings

    def generate_instance_input(self, table):
        col_embeddings = [
            self._transform(table[c].tolist())
            for c in table.columns
            #if not is_numeric(t1[c]) and t1[c].count() > 0
        ]

        col_embeddings = np.stack(col_embeddings)
        return col_embeddings

    def generate_column_input(self, column_values):
        if self.num_rows > 0 and self.num_rows < len(column_values):
            column_values = random.sample(column_values, self.num_rows)

        col_embedding = self._transform(column_values)
        col_embedding = torch.as_tensor(col_embedding, dtype=torch.float32)

        return col_embedding
    # def collate(self, batch):
    #     t_embeddings = []
    #     for t_col_embeddings in batch:
    #         t_embeddings.extend(t_col_embeddings)
        
    #     t_embeddings = np.array(t_embeddings)
    #     t_embeddings = torch.as_tensor(t_embeddings, dtype=torch.float32)
    #     return t_embeddings


def collate(pairs): # [batch size, (number of columns, embedding dimension; -)]
    # print(type(pairs)) # list
    # print(type(pairs[0])) # tuple
    # batch_size = len(pairs)
    # embed_dim = pairs[0][0].shape[1]

    t1_embeddings, t2_embeddings = [], []
    for t1_col_embeddings, t2_col_embeddings in pairs:
        t1_embeddings.extend(t1_col_embeddings)
        t2_embeddings.extend(t2_col_embeddings)
    
    t1_embeddings = torch.as_tensor(np.stack(t1_embeddings), dtype=torch.float32)
    t2_embeddings = torch.as_tensor(np.stack(t2_embeddings), dtype=torch.float32)

    return (t1_embeddings, t2_embeddings)


def instance_collate(batch):
    t_embeddings = []
    for t_col_embeddings in batch:
        t_embeddings.extend(t_col_embeddings)
    
    t_embeddings = torch.as_tensor(np.stack(t_embeddings), dtype=torch.float32)
    return t_embeddings


class GitTablesDataset(Dataset):
    def __init__(self, table_dir, table_roster, read_csv=False):
        super().__init__()

        self.table_dir = table_dir
        self.table_roster = table_roster
        self.read_csv = read_csv
        self.X = self._load_tables()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def _load_tables(self):
        print("Start Loading GitTables...")
        X = []
        with open(self.table_roster, 'r') as f:
            table_names = f.readlines()

            for tn in table_names:
                table_path = os.path.join(self.table_dir, tn.rstrip())
                if self.read_csv:
                    table = pd.read_csv(table_path, sep=",", lineterminator="\n")
                    table = table.select_dtypes(include="object") # only consider textual columns
                    if table.empty: continue
                else:
                    table = pq.read_table(table_path).to_pandas()
                X.append(table)

        print("Finish Loading GitTables...")
        return X


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, tbl_ds, model_path, num_rows=5):
       super().__init__()

       self.tbl_ds = tbl_ds
       self.input_formatter = InputFormatter(model_path, num_rows=num_rows)
    
    def __len__(self):
        return len(self.tbl_ds)

    def __getitem__(self, idx):  
        table = self.tbl_ds[idx]
        t1_input, t2_input = self.input_formatter.generate_input_pairs(table, idx)

        return t1_input, t2_input


class TestDatasetWrapper(Dataset):
    def __init__(self, tbl_ds, model_path, num_rows=3):
       super().__init__()

       self.tbl_ds = tbl_ds
       self.input_formatter = InputFormatter(model_path, num_rows=num_rows)
    
    def __len__(self):
        return len(self.tbl_ds)

    def __getitem__(self, idx):  
        table = self.tbl_ds[idx]
        t_input = self.input_formatter.generate_instance_input(table)

        return t_input
