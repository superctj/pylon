# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import gc
import math
import sys
import numpy as np
import torch
import torch.nn as nn

from pprint import pprint
from typing import Dict, Tuple
# from fairseq import distributed_utils
from torch_scatter import scatter_mean
from tqdm import tqdm
from transformers import BertModel
from transformers.models.bert.modeling_bert import (
    BertSelfOutput, BertIntermediate, BertOutput
)
from TaBERT.vanilla_table_bert import VanillaTableBert, TableBertConfig
from TaBERT.vertical_attention_config import VerticalAttentionTableBertConfig


class VerticalEmbeddingLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states


class BertVerticalAttention(nn.Module):
    def __init__(self, config: TableBertConfig):
        nn.Module.__init__(self)

        self.self_attention = VerticalSelfAttention(config)
        self.self_output = BertSelfOutput(config) # linear transformation + dropout + layer normalization

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        output = self.self_output(self_attention_output, hidden_states)

        return output


class VerticalSelfAttention(nn.Module):
    def __init__(self, config: TableBertConfig):
        super(VerticalSelfAttention, self).__init__()

        if config.hidden_size % config.num_vertical_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_vertical_attention_heads))

        self.num_attention_heads = config.num_vertical_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_vertical_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_linear = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        # (batch_size, max_sequence_len, num_attention_heads, max_row_num, attention_head_size)
        x = x.permute(0, 2, 3, 1, 4)

        return x

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        mixed_query_layer = self.query_linear(hidden_states)
        mixed_key_layer = self.key_linear(hidden_states)
        mixed_value_layer = self.value_linear(hidden_states)

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, max_row_num)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # TODO: consider remove this cell dropout?

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 3, 1, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertVerticalLayer(nn.Module):
    def __init__(self, config: VerticalAttentionTableBertConfig):
        nn.Module.__init__(self)

        self.attention = BertVerticalAttention(config)
        self.intermediate = BertIntermediate(config) # non-linear transformation
        self.output = BertOutput(config) # linear transformation + dropout + layer normalization

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VerticalAttentionTableBert(VanillaTableBert):
    CONFIG_CLASS = VerticalAttentionTableBertConfig

    def __init__(
        self,
        config: VerticalAttentionTableBertConfig,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, **kwargs)

        self._bert_model = BertModel.from_pretrained(config.base_model_name)
        for param in self._bert_model.parameters():
            param.requires_grad = False
        self.vertical_embedding_layer = VerticalEmbeddingLayer()
        self.vertical_transformer_layers = nn.ModuleList([
            BertVerticalLayer(self.config)
            for _ in range(self.config.num_vertical_layers)
        ])

        # if config.initialize_from:
        #     print(f'Loading initial parameters from {config.initialize_from}', file=sys.stderr)
        #     initial_state_dict = torch.load(config.initialize_from, map_location='cpu')
        #     if not any(key.startswith('_bert_model') for key in initial_state_dict):
        #         print('warning: loading model from an old version', file=sys.stderr)
        #         bert_model = BertModel.from_pretrained(
        #             config.base_model_name,
        #             state_dict=initial_state_dict
        #         )
        #         self._bert_model = bert_model
        #     else:
        #         load_result = self.load_state_dict(initial_state_dict, strict=False)
        #         if load_result.missing_keys:
        #             print(f'warning: missing keys: {load_result.missing_keys}', file=sys.stderr)
        #         if load_result.unexpected_keys:
        #             print(f'warning: unexpected keys: {load_result.unexpected_keys}', file=sys.stderr)

        added_modules = [self.vertical_embedding_layer, self.vertical_transformer_layers]

        for module in added_modules:
            module.apply(self._bert_model._init_weights)

    @property
    def parameter_type(self):
        return next(self.parameters()).dtype

    # def forward(
    #     self,
    #     bert_output,
    #     cell_token_col_ids_expanded,
    #     table_mask,
    #     **kwargs
    # ):
    #     # """

    #     # Args:
    #     #     input_ids: (batch_size, max_row_num, sequence_len)
    #     #     segment_ids: (batch_size, max_row_num, sequence_len)
    #     #     cell_token_col_ids: (batch_size, max_row_num, sequence_len)
    #     #     sequence_mask: (batch_size, max_row_num, sequence_len)
    #     #     table_mask: (batch_size, max_row_num, max_column_num)
    #     # """

    #     max_col_num = table_mask.size(-1)
    #     table_embedding = scatter_mean(
    #         src=bert_output,
    #         index=cell_token_col_ids_expanded,
    #         dim=-2,  # over `sequence_len`
    #         dim_size=max_col_num + 1   # last dimension is used for collecting special tokens in a sequence like [SEP] and [PAD]
    #     )
    #     # print("Table encoding shape: ", table_embedding.shape)
    #     table_embedding = table_embedding[:, :, :-1, :] * table_mask.unsqueeze(-1)
    #     # print(table_embedding[:, :, -1, :])
    #     # print("Table embedding shape before vertical attention: ", table_embedding.shape)

    #     # perform vertical attention
    #     table_embedding = self.vertical_transform(table_embedding, table_mask)
    #     return table_embedding
    
    # noinspection PyMethodOverriding
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        cell_token_col_ids: torch.Tensor,
        sequence_mask: torch.Tensor,
        table_mask: torch.Tensor,
        **kwargs
    ):
        """

        Args:
            input_ids: (batch_size, max_row_num, sequence_len)
            segment_ids: (batch_size, max_row_num, sequence_len)
            cell_token_col_ids: (batch_size, max_row_num, sequence_len)
            sequence_mask: (batch_size, max_row_num, sequence_len)
            table_mask: (batch_size, max_row_num, max_column_num)
        """

        batch_size, max_row_num, sequence_len = input_ids.size()

        # if self.parameter_type == torch.float16:
        #     sequence_mask = sequence_mask.to(dtype=torch.float16)
        #     context_token_mask = context_token_mask.to(dtype=torch.float16)
        #     table_mask = table_mask.to(dtype=torch.float16)

        flattened_input_ids = input_ids.view(batch_size * max_row_num, -1)
        flattened_segment_ids = segment_ids.view(batch_size * max_row_num, -1)
        flattened_sequence_mask = sequence_mask.view(batch_size * max_row_num, -1)

        # (batch_size * max_row_num, sequence_len, hidden_size)
        # (sequence_output, pooler_output)
        kwargs = {}

        bert_output = self._bert_model(
            input_ids=flattened_input_ids,
            token_type_ids=flattened_segment_ids,
            attention_mask=flattened_sequence_mask,
            **kwargs
        )
        bert_output = bert_output.last_hidden_state
        bert_output = bert_output.view(batch_size, max_row_num, sequence_len, -1) # (batch_size, max_row_num, sequence_len, hidden_size)

        # expand to the same size as `bert_output`
        # print('cell_token_col_ids shape: ', cell_token_col_ids.shape)
        cell_token_col_id_expanded = cell_token_col_ids.unsqueeze(-1).expand(
            -1, -1, -1, bert_output.size(-1)) # (batch_size, max_row_num, sequence_len, hidden_size)
        # print('cell_token_col_id_expanded shape: ', cell_token_col_id_expanded.shape)

        # (batch_size, max_row_num, max_column_num, hidden_size)
        max_col_num = table_mask.size(-1)
        table_embedding = scatter_mean(
            src=bert_output,
            index=cell_token_col_id_expanded,
            dim=-2, # over `sequence_len`
            dim_size=max_col_num + 1 # last dimension is used for collecting special tokens in a sequence like [SEP] and [PAD]
        )
        # print("Table encoding shape: ", table_embedding.shape)
        table_embedding = table_embedding[:, :, :-1, :] * table_mask.unsqueeze(-1)
        # print(table_embedding[:, :, -1, :])
        # print("Table embedding shape before vertical attention: ", table_embedding.shape)

        # perform vertical attention
        table_embedding = self.vertical_transform(table_embedding, table_mask)
        return table_embedding

    def vertical_transform(self, table_encoding, table_mask):
        # (batch_size, sequence_len, 1, max_row_num, 1)
        attention_mask = table_mask.permute(0, 2, 1)[:, :, None, :, None]
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = table_encoding
        vertical_layer_outputs = []
        for vertical_layer in self.vertical_transformer_layers:
            hidden_states = vertical_layer(hidden_states, attention_mask=attention_mask)
            vertical_layer_outputs.append(hidden_states)

        last_hidden_states = vertical_layer_outputs[-1] * table_mask.unsqueeze(-1)

        # mean-pool last encoding
        # (batch_size, 1, 1)
        table_row_nums = table_mask[:, :, 0].sum(dim=-1)[:, None, None]
        # (batch_size, max_column_num, hidden_size)
        mean_pooled_table_encoding = last_hidden_states.sum(dim=1) / table_row_nums

        return mean_pooled_table_encoding

    # def encode(
    #         self,
    #         table_tensor_dict,
    #         return_bert_encoding: bool = False
    # ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    #     assert return_bert_encoding is False, 'VerticalTableBert does not support `return_bert_encoding=True`'

        # tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        # tensor_dict = {
        #     k: v.to(self.device) if torch.is_tensor(v) else v
        #     for k, v in tensor_dict.items()
        # }

        # table_embedding = self.forward(**table_tensor_dict)
        # return table_embedding

        # tensor_dict['context_token_mask'] = tensor_dict['context_token_mask'][:, 0, :]
        # tensor_dict['column_mask'] = tensor_dict['table_mask'][:, 0, :]

        # info = {
        #     'tensor_dict': tensor_dict,
        #     'instances': instances
        # }

        # return context_encoding, schema_encoding, info

if __name__ == '__main__':
    table_bert_config = VerticalAttentionTableBertConfig()
    pprint(table_bert_config)
    # pprint(table_bert_config.hidden_size)
    tabert_encoder = VerticalAttentionTableBert(table_bert_config)
    for param_tensor in tabert_encoder.state_dict():
        print(param_tensor, "\t", tabert_encoder.state_dict()[param_tensor].size())