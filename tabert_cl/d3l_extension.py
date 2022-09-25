import os

from typing import Iterable, Set, Optional, Dict

import numpy as np
import pandas as pd

from transformers import BertTokenizer

from tabert_cl_training.input_prep import InputFormatter
from util import load_pylon_tabert_model


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
        tbl_tensor_dict = self._input_formatter.collate([tbl_input], row_num=self._num_samples)

        # Get projected embeddings
        _, embeddings = self._embedding_model.inference(tbl_tensor_dict)
        # print("Embedding shape: ", embeddings.shape)
        # exit()
        embeddings = embeddings[0].detach().cpu().numpy()
        return embeddings
