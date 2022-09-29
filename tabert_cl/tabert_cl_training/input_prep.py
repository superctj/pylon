import os

from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer


MAX_BERT_INPUT_LENGTH = 512
MAX_CELL_LEN = 30
CELL_FORMAT = ["value"]
COL_DELIMITER = "[SEP]"
TRIM_LONG_TABLE = True


class TableTooLongError(ValueError):
    pass


class InputFormatter(object):
    def __init__(self, tokenizer: BertTokenizer, num_rows: int):
        self.tokenizer = tokenizer
        self.num_rows = num_rows

    def generate_input_pairs(self, table: pd.DataFrame) -> Tuple:    
        t1 = self._sample_rows(table, self.num_rows)
        t2 = self._sample_rows(table, self.num_rows)
        t1_header, t1_tokenized = self._tokenize_table(t1)
        t2_header, t2_tokenized = self._tokenize_table(t2)
        # pprint(t1_header)
        # pprint(t2_header)
        # pprint(t1_tokenized)
        # pprint(t2_tokenized)
        t1_input = self._get_table_input(t1_header, t1_tokenized)
        t2_input = self._get_table_input(t2_header, t2_tokenized)
        # pprint(t1_input)
        # pprint(t2_input)
        return t1_input, t2_input

    def _sample_rows(self, table: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        return table.sample(n=num_rows) if table.shape[0] > num_rows else table
    
    def _tokenize_table(self, table: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
        header = list(table.columns)
        tokenized_data = []

        # Tokenize rows
        for i in range(table.shape[0]):
            tokenized_row = {}
            for col_name in header:
                tokenized_row[col_name] = self.tokenizer.tokenize(str(table.iloc[i][col_name]).lower())
            tokenized_data.append(tokenized_row)

        # pprint(header)
        # pprint(tokenized_header)
        # pprint(tokenized_data)
        # return header, (tokenized_header, tokenized_data)
        return header, tokenized_data
    
    def _get_table_input(self, header: List[str], tokenized_data: Tuple[Dict, List[Dict]]) -> List[Dict]:
        # pprint(header)
        # print("Number of columns: ", len(header))
        # pprint(tokenized_data)
        table_input = []
        for row_data in tokenized_data:
            row_input = self._get_row_input(row_data, header)
            table_input.append(row_input)

        return table_input
    
    def _get_row_input(self, row: Dict, header: List[str]) -> Dict:
        # Account for [CLS] token at the beginning and [SEP] token at the end
        row_token_max_len = MAX_BERT_INPUT_LENGTH - 2
        row_input_tokens = []
        row_token_span_map = []
        cell_start_idx = 1 # Account for [CLS] at the beginning

        for i, col_name in enumerate(header):
            if i >= 20: break # Consider at most 20 columns

            value_tokens = row[col_name]
            # if len(value_tokens) == 0:
            #     print(header)
            #     print(row)
            #     raise ValueError("Empty cell!")
            value_tokens = value_tokens[:MAX_CELL_LEN] # Truncate long cells

            cell_input_tokens, cell_span_map = self._get_cell_input(value_tokens, cell_start_idx)
            cell_input_tokens.append(COL_DELIMITER)

            early_stop = False
            if TRIM_LONG_TABLE:
                if len(row_input_tokens) + len(cell_input_tokens) > row_token_max_len:
                    valid_cell_input_token_len = row_token_max_len - len(row_input_tokens)
                    cell_input_tokens = cell_input_tokens[:valid_cell_input_token_len]
                    end_index = cell_start_idx + len(cell_input_tokens)
                    
                    keys_to_delete = []
                    for key in cell_span_map:
                        if key in {"column_name", "type", "value", "whole_span"}:
                            span_start_idx, span_end_idx = cell_span_map[key]
                            if span_start_idx < end_index < span_end_idx:
                                cell_span_map[key] = (span_start_idx, end_index)
                            elif end_index < span_start_idx:
                                keys_to_delete.append(key)
                        
                        elif key == "delimiter":
                            old_positions = cell_span_map[key]
                            new_positions = [idx for idx in old_positions if idx < end_index]
                            if not new_positions:
                                keys_to_delete.append(key)

                    for key in keys_to_delete:
                        del cell_span_map[key]

                    # nothing left, we just skip this cell and break
                    if len(cell_span_map) == 0:
                        break
                    early_stop = True
                
                elif len(row_input_tokens) + len(cell_input_tokens) == row_token_max_len: early_stop = True
            
            elif len(row_input_tokens) + len(cell_input_tokens) > row_token_max_len: break

            row_input_tokens.extend(cell_input_tokens)
            cell_start_idx = cell_start_idx + len(cell_input_tokens)
            row_token_span_map.append(cell_span_map)

            if early_stop: break
        
        # It is possible that the first cell is too long and cannot fit into `row_token_max_len` and we need to discard this table
        if len(row_input_tokens) == 0:
            pprint(f"Header: {header}")
            pprint(f"Row: {row}")
            raise TableTooLongError()

        if row_input_tokens[-1] == COL_DELIMITER:
            row_input_tokens = row_input_tokens[:-1]

        row_input_tokens = ["[CLS]"] + row_input_tokens + ["[SEP]"]
        row_instance = {
            # "header": header,
            "tokens": row_input_tokens,
            "token_ids": self.tokenizer.convert_tokens_to_ids(row_input_tokens),
            "cell_spans": row_token_span_map
        }

        row_input_token_len = len(row_input_tokens)
        # Specify to which column a cell token belongs
        cell_token_col_ids = [np.iinfo(np.uint16).max] * row_input_token_len

        for col_id, col_name in enumerate(header):
            if col_id < len(row_instance["cell_spans"]):
                col_start, col_end = row_instance["cell_spans"][col_id]["whole_span"]
                cell_token_col_ids[col_start:col_end] = [col_id] * (col_end - col_start)

        row_instance["cell_token_col_ids"] = cell_token_col_ids
        return row_instance

    def _get_cell_input(self, cell_tokens: List[str], cell_start_idx: int) -> Tuple[List[str], Dict]:
        # The orginal cell could be a string containing only whitespace
        if len(cell_tokens) == 0:
            cell_tokens.append("-")
        
        span_map = {}
        span_map["value"] = (cell_start_idx, cell_start_idx + len(cell_tokens))
        span_map["whole_span"] = (cell_start_idx, cell_start_idx + len(cell_tokens))

        return cell_tokens, span_map
    
    def generate_instance_input(self, table: pd.DataFrame) -> List[Dict]:
        # table = table.select_dtypes(include="object")
        # table = table.dropna(axis="columns", how="all") # Drop empty columns
        # table = table.dropna(axis="index", how="any") # Drop rows with missing values; use "all" for tus benchmark
        # # table = table.dropna(axis="index", how="all") # drop rows with missing values
        # if table.shape[1] < 1:
        #     print(f"Table has no column left...")
        # if table.shape[0] < 3:
        #     print(f"Table has fewer than 3 rows...")
        
        t_sample = self._sample_rows(table, self.num_rows)
        t_header, t_tokenized = self._tokenize_table(t_sample)
        t_input = self._get_table_input(t_header, t_tokenized)

        return t_input
    
    def collate(self, tables: List[List]) -> Dict:
        batch_size = len(tables)
        max_seq_len = 0

        if batch_size == 1: # for inference
            row_num = min(self.num_rows, len(tables[0]))

        for tbl in tables:
            local_max = max(len(row["token_ids"]) for row in tbl)
            max_seq_len = max(max_seq_len, local_max)

        t_input_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
        t_seq_mask = np.zeros((batch_size, row_num, max_seq_len), dtype=np.float32)
        t_segment_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
        t_row_col_nums = []

        # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
        cell_token_col_ids_fill_val = np.iinfo(np.uint16).max
        t_cell_token_col_ids = np.full((batch_size, row_num, max_seq_len), dtype=int, fill_value=cell_token_col_ids_fill_val)

        for idx, t in enumerate(tables):
            for row_id, row_inst in enumerate(t):
                bert_input_seq_len = len(row_inst["token_ids"])
                t_input_ids[idx, row_id, :bert_input_seq_len] = row_inst["token_ids"]
                t_seq_mask[idx, row_id, :bert_input_seq_len] = 1
                t_segment_ids[idx, row_id, :bert_input_seq_len] = 1

                row_cell_token_col_ids = np.array(row_inst["cell_token_col_ids"])
                cur_col_num = row_cell_token_col_ids[row_cell_token_col_ids != cell_token_col_ids_fill_val].max() + 1
                t_row_col_nums.append(cur_col_num)
                t_cell_token_col_ids[idx, row_id, :len(row_cell_token_col_ids)] = row_cell_token_col_ids

        max_col_num = max(t_row_col_nums)
        t_table_mask = np.zeros((batch_size, row_num, max_col_num), dtype=int)
        t_global_col_idx = 0

        for idx, t in enumerate(tables):
            for row_id, _ in enumerate(t):
                col_num = t_row_col_nums[t_global_col_idx]
                t_table_mask[idx, row_id, :col_num] = 1
                t_global_col_idx += 1

        t_cell_token_col_ids[t_cell_token_col_ids == cell_token_col_ids_fill_val] = max_col_num

        t_tensor_dict = {
            "input_ids": torch.tensor(t_input_ids, dtype=torch.long),
            "segment_ids": torch.tensor(t_segment_ids, dtype=torch.long),
            "cell_token_col_ids": torch.tensor(t_cell_token_col_ids, dtype=torch.long),
            "sequence_mask": torch.tensor(t_seq_mask, dtype=torch.float32),
            "table_mask": torch.tensor(t_table_mask, dtype=torch.float32)
        }

        return t_tensor_dict


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
        with open(self.table_roster, "r") as f:
            table_names = f.readlines()

            for tn in table_names:
                table_path = os.path.join(self.table_dir, tn.rstrip())
                if self.read_csv:
                    table = pd.read_csv(table_path, sep=",", lineterminator="\n")
                else:
                    table = pq.read_table(table_path).to_pandas()
                
                table = table.select_dtypes(include="object") # only consider textual columns
                if table.empty: continue

                X.append(table)

        print("Finish Loading GitTables...")
        return X


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, tbl_ds, num_rows=3):
       super().__init__()

       self.tbl_ds = tbl_ds
       self.input_formatter = InputFormatter(BertTokenizer.from_pretrained("bert-base-uncased"), num_rows)
    
    def __len__(self):
        return len(self.tbl_ds)

    def __getitem__(self, idx):  
        table = self.tbl_ds[idx]
        t1_input, t2_input = self.input_formatter.generate_input_pairs(table, idx)

        return t1_input, t2_input


class TestDatasetWrapper(Dataset):
    def __init__(self, tbl_ds, num_rows=3):
       super().__init__()

       self.tbl_ds = tbl_ds
       self.input_formatter = InputFormatter(BertTokenizer.from_pretrained("bert-base-uncased"), num_rows)
    
    def __len__(self):
        return len(self.tbl_ds)

    def __getitem__(self, idx):  
        table = self.tbl_ds[idx]
        t_input = self.input_formatter.generate_instance_input(table)

        return t_input


def collate(table_pairs, row_num=5): # list of tuples (t1, t2)
    batch_size = len(table_pairs)
    max_seq_len = 0

    for t1, t2 in table_pairs:
        # assert t1[1]["header"] == t2[2]["header"]
        t1_local_max = max(len(row["token_ids"]) for row in t1)
        t2_local_max = max(len(row["token_ids"]) for row in t2)
        max_seq_len = max(max_seq_len, t1_local_max, t2_local_max)
    # print("Max sequence length: ", max_seq_len)

    t1_input_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
    t2_input_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
    t1_seq_mask = np.zeros((batch_size, row_num, max_seq_len), dtype=np.float32)
    t2_seq_mask = np.zeros((batch_size, row_num, max_seq_len), dtype=np.float32)
    t1_segment_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
    t2_segment_ids = np.zeros((batch_size, row_num, max_seq_len), dtype=np.int64)
    t1_row_col_nums = []
    t2_row_col_nums = []

    # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
    cell_token_col_ids_fill_val = np.iinfo(np.uint16).max
    t1_cell_token_col_ids = np.full((batch_size, row_num, max_seq_len), dtype=int, fill_value=cell_token_col_ids_fill_val)
    t2_cell_token_col_ids = np.full((batch_size, row_num, max_seq_len), dtype=int, fill_value=cell_token_col_ids_fill_val)

    for idx, (t1, t2) in enumerate(table_pairs):
        for row_id, row_inst in enumerate(t1):
            bert_input_seq_len = len(row_inst["token_ids"])
            t1_input_ids[idx, row_id, :bert_input_seq_len] = row_inst["token_ids"]
            t1_seq_mask[idx, row_id, :bert_input_seq_len] = 1
            t1_segment_ids[idx, row_id, :bert_input_seq_len] = 1

            row_cell_token_col_ids = np.array(row_inst["cell_token_col_ids"])
            try:
                cur_col_num = row_cell_token_col_ids[row_cell_token_col_ids != cell_token_col_ids_fill_val].max() + 1
            except ValueError:
                print(row_cell_token_col_ids)
                raise ValueError
            t1_row_col_nums.append(cur_col_num)
            t1_cell_token_col_ids[idx, row_id, :len(row_cell_token_col_ids)] = row_cell_token_col_ids
        
        for row_id, row_inst in enumerate(t2):
            bert_input_seq_len = len(row_inst["token_ids"])
            t2_input_ids[idx, row_id, :bert_input_seq_len] = row_inst["token_ids"]
            t2_seq_mask[idx, row_id, :bert_input_seq_len] = 1
            t2_segment_ids[idx, row_id, :bert_input_seq_len] = 1

            row_cell_token_col_ids = np.array(row_inst["cell_token_col_ids"])
            try:
                cur_col_num = row_cell_token_col_ids[row_cell_token_col_ids != cell_token_col_ids_fill_val].max() + 1
            except ValueError:
                print(row_cell_token_col_ids)
                raise ValueError
            t2_row_col_nums.append(cur_col_num)
            t2_cell_token_col_ids[idx, row_id, :len(row_cell_token_col_ids)] = row_cell_token_col_ids

    t1_max_col_num = max(t1_row_col_nums)
    t2_max_col_num = max(t2_row_col_nums)
    max_col_num = max(t1_max_col_num, t2_max_col_num)
    # print("Maximum column number: ", max_col_num)
    
    t1_table_mask = np.zeros((batch_size, row_num, max_col_num), dtype=int)
    t2_table_mask = np.zeros((batch_size, row_num, max_col_num), dtype=int)
    t1_global_col_idx, t2_global_col_idx = 0, 0

    for idx, (t1, t2) in enumerate(table_pairs):
        for row_id, _ in enumerate(t1):
            col_num = t1_row_col_nums[t1_global_col_idx]
            t1_table_mask[idx, row_id, :col_num] = 1
            t1_global_col_idx += 1
        for row_id, _ in enumerate(t2):
            col_num = t2_row_col_nums[t2_global_col_idx]
            t2_table_mask[idx, row_id, :col_num] = 1
            t2_global_col_idx += 1

    t1_cell_token_col_ids[t1_cell_token_col_ids == cell_token_col_ids_fill_val] = t1_max_col_num
    t2_cell_token_col_ids[t2_cell_token_col_ids == cell_token_col_ids_fill_val] = t2_max_col_num

    t1_tensor_dict = {
        "input_ids": torch.tensor(t1_input_ids, dtype=torch.long),
        "segment_ids": torch.tensor(t1_segment_ids, dtype=torch.long),
        "cell_token_col_ids": torch.tensor(t1_cell_token_col_ids, dtype=torch.long),
        "sequence_mask": torch.tensor(t1_seq_mask, dtype=torch.float32),
        "table_mask": torch.tensor(t1_table_mask, dtype=torch.float32)
    }

    t2_tensor_dict = {
        "input_ids": torch.tensor(t2_input_ids, dtype=torch.long),
        "segment_ids": torch.tensor(t2_segment_ids, dtype=torch.long),
        "cell_token_col_ids": torch.tensor(t2_cell_token_col_ids, dtype=torch.long),
        "sequence_mask": torch.tensor(t2_seq_mask, dtype=torch.float32),
        "table_mask": torch.tensor(t2_table_mask, dtype=torch.float32)
    }

    return (t1_tensor_dict, t2_tensor_dict)