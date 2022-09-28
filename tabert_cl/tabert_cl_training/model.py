import torch
import torch.nn as nn
import torch.nn.functional as F
from TaBERT.vertical_attention_config import VerticalAttentionTableBertConfig
from TaBERT.vertical_attention_table_bert import VerticalAttentionTableBert


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()

        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb_i, emb_j):
        # emb_i shape: (number of columns, projected embedding dimension)
        # assert(len(emb_i.shape) == 2)
        # assert(emb_i.shape == emb_j.shape)
        span = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

        sim_ij = torch.diag(similarity_matrix, span)
        sim_ji = torch.diag(similarity_matrix, -span)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)

        negatives_mask = ~torch.eye(span * 2, span * 2, dtype=bool, device=emb_i.device)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * span)

        # self_mask = ~negatives_mask
        # similarity_matrix.masked_fill_(self_mask, -9e15)
        # pos_mask = self_mask.roll(shifts=span, dims=-1)
        # comb_sim = torch.cat(
        #     [similarity_matrix[pos_mask][:, None], similarity_matrix.masked_fill(pos_mask, -9e15)],
        #     dim=-1,
        # )

        # sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # acc = (sim_argsort == 0).float().mean()

        # return loss, acc
        return loss


class SimTableCLR(nn.Module):
    def __init__(self, projection_size):
        super().__init__()

        table_bert_config = VerticalAttentionTableBertConfig()
        self.encoder = VerticalAttentionTableBert(table_bert_config)
        self.projection = nn.Sequential(
            nn.Linear(in_features=table_bert_config.hidden_size, out_features=projection_size),
            nn.ReLU(),
            nn.Linear(in_features=projection_size, out_features=projection_size)
        )
    
    def get_tabert_encoding(self, table_tensor_dict):
        return self.encoder(**table_tensor_dict)

    def forward(self, table_tensor_dict_pair):
        t1_tensor_dict, t2_tensor_dict = table_tensor_dict_pair

        # (batch size, max number of columns, embedding dimension)
        t1_embedding = self.get_tabert_encoding(t1_tensor_dict)
        t2_embedding = self.get_tabert_encoding(t2_tensor_dict)

        # Remove all-zero dummy columns to save memory in loss calculation
        # Need to consider both masks as t1 and t2 may differ in column numbers due to different cell lengths
        t1_non_zero_cols = torch.abs(t1_embedding).sum(dim=2) > 0
        t2_non_zero_cols = torch.abs(t2_embedding).sum(dim=2) > 0
        non_zero_cols = torch.logical_and(t1_non_zero_cols, t2_non_zero_cols)

        # (number of columns, embedding dimension)
        t1_embedding = t1_embedding[non_zero_cols]
        t2_embedding = t2_embedding[non_zero_cols]
        
        t1_projection = self.projection(t1_embedding)
        t2_projection = self.projection(t2_embedding)
        return t1_projection, t2_projection
    
    def inference(self, table_tensor_dict):
        embedding = self.get_tabert_encoding(table_tensor_dict)
        projection = self.projection(embedding)
        return embedding, projection
