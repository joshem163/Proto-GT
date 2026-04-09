import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def edge_weights_from_similarity(x_sim: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 tau: float = 0.5,
                                 eps: float = 1e-12) -> torch.Tensor:
    """
    Compute per-edge weights from cosine similarity of node embeddings x_sim,
    then normalize weights with a softmax over incoming edges of each target node.

    edge_index: [2, E] with edge_index[0]=src, edge_index[1]=dst
    returns: edge_weight [E] where sum_{src->dst} edge_weight = 1 for each dst
    """
    src, dst = edge_index

    # cosine similarity on edges
    x_i = x_sim[src]
    x_j = x_sim[dst]
    x_i = x_i / (x_i.norm(p=2, dim=-1, keepdim=True) + eps)
    x_j = x_j / (x_j.norm(p=2, dim=-1, keepdim=True) + eps)
    sim = (x_i * x_j).sum(dim=-1)  # [E] in [-1, 1]

    # temperature + normalize across incoming edges per dst
    logits = sim / max(tau, eps)
    edge_weight = softmax(logits, dst)  # [E], sums to 1 for each dst
    return edge_weight


class WeightedSAGEConv(MessagePassing):
    """
    GraphSAGE-style conv, but with similarity-based (normalized) edge weights.
    Aggregation is a weighted sum over neighbors (weights sum to 1 per target),
    so it acts like a weighted mean.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')  # weighted mean via normalized weights
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root  = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.lin_neigh.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        # x: [N, F], edge_weight: [E] normalized per dst
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_neigh(out) + self.lin_root(x)
        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # x_j: neighbor features at src nodes for each edge
        return x_j * edge_weight.view(-1, 1)


class SAGE_SimWeighted(torch.nn.Module):
    """
    Same structure as your SAGE model, but uses similarity-based edge weights.

    forward(x, edge_index, x_sim=None):
      - x is what you want to classify on (raw, new, fused, etc.)
      - x_sim is the embedding used to compute similarity weights (often your new embedding).
        If x_sim is None, it uses x.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 tau: float = 0.5):
        super().__init__()

        self.tau = tau
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(WeightedSAGEConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(WeightedSAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(WeightedSAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, x_sim=None):
        # x_sim: the embedding you trust for similarity (e.g., your novel embedding)
        if x_sim is None:
            x_sim = x

        # compute once from x_sim (you can also recompute per-layer using updated h if you want)
        edge_weight = edge_weights_from_similarity(x_sim, edge_index, tau=self.tau)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return x
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d