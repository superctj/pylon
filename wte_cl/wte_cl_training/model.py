import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer('negatives_mask', (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        # emb_i shape: (number of columns, projected embedding dimension)
        span = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

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

        return loss


class SimTableCLR(nn.Module):
    def __init__(self, embedding_dim, projection_size):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=projection_size),
            nn.ReLU(),
            nn.Linear(in_features=projection_size, out_features=projection_size)
        )

    def forward(self, embedding):
        projection = self.projection(embedding)
        return projection