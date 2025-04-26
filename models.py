
import torch
import torch.nn.functional as F
import networkx as nx
from networkx import ego_graph
import torch.nn as nn
import torch.nn.functional as F
#from torch_sparse import SparseTensor,matmul
import scipy.sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch.optim as optim
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv,GATConv, JumpingKnowledge,TransformerConv


#from logger import Logger
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Final layer (concat=False for classification output)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters_mlp(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            # x = F.relu(x)
            x = F.sigmoid(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # return torch.log_softmax(x, dim=-1)
        return x


class MLP2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP2, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters_mlp2(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            # x = F.relu(x)
            x = F.sigmoid(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class TWGNN(nn.Module):
    def __init__(self, gnn_in, mlp_in, hidden_dim, fusion_dim, num_classes,num_layers,dropout):
        super(TWGNN, self).__init__()

        # GraphSAGE for original features
        self.graphsage = SAGE(gnn_in, hidden_dim, 3*fusion_dim,num_layers,dropout)

        # MLP for another feature set
        self.mlp = MLP(mlp_in, hidden_dim, fusion_dim,num_layers,dropout)

        # Learnable weights for linear combination
        self.alpha = nn.Parameter(torch.rand(1))  # Weight for GraphSAGE output
        self.beta = nn.Parameter(torch.rand(1))  # Weight for MLP output

        # Final classifier (small neural network)
        self.classifier = nn.Sequential(
            nn.Linear(4*fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def reset_parameters(self):
        """ Reset parameters for all components. """
        self.graphsage.reset_parameters()
        self.mlp.reset_parameters_mlp()
        nn.init.uniform_(self.alpha, 0, 1)  # Reinitialize learnable weights
        nn.init.uniform_(self.beta, 0, 1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    def forward(self, x_gsage, edge_index, x_mlp):
        h_gsage = self.graphsage(x_gsage, edge_index)  # Graph-based embeddings
        h_mlp = self.mlp(x_mlp)  # MLP-based embeddings

        # Linear combination
        # h_fused = self.alpha * h_gsage + self.beta * h_mlp
        #h_fused = self.alpha * h_gsage + (1-self.alpha)* h_mlp
        h_fused = torch.cat([h_gsage, h_mlp], dim=1)

        # Final classification
        out = self.classifier(h_fused)
        return out


class TW_TF(nn.Module):
    def __init__(self, gnn_in, mlp_in, hidden_dim, fusion_dim, num_classes,num_layers,dropout):
        super(TW_TF, self).__init__()

        # GraphSAGE for original features
        self.graphTF = TransformerConv(gnn_in,fusion_dim,heads=2,dropout=0.5)

        # MLP for another feature set
        self.mlp = MLP(mlp_in, hidden_dim, 2*fusion_dim,num_layers,dropout)

        # Learnable weights for linear combination
        self.alpha = nn.Parameter(torch.rand(1))  # Weight for GraphSAGE output
        self.beta = nn.Parameter(torch.rand(1))  # Weight for MLP output

        # Final classifier (small neural network)
        self.classifier = nn.Sequential(
            nn.Linear(2*fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def reset_parameters(self):
        """ Reset parameters for all components. """
        self.graphTF.reset_parameters()
        self.mlp.reset_parameters_mlp()
        nn.init.uniform_(self.alpha, 0, 1)  # Reinitialize learnable weights
        nn.init.uniform_(self.beta, 0, 1)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    def forward(self, x_gsage, edge_index, x_mlp):
        h_gsage = self.graphTF(x_gsage, edge_index)  # Graph-based embeddings
        h_mlp = self.mlp(x_mlp)  # MLP-based embeddings

        # Linear combination
        h_fused = self.alpha * h_gsage + self.beta * h_mlp

        # Final classification
        out = self.classifier(h_fused)
        return out



def train(model, mlp_model, mlp_2, data, train_idx, optimizer, optimizer_mlp, optimizer_mlp2):
    model.train()
    mlp_model.train()
    mlp_2.train()
    optimizer.zero_grad()
    optimizer_mlp.zero_grad()
    optimizer_mlp2.zero_grad()
    gcn_embedding = model(data.x, data.adj_t)[train_idx]
    mlp_embedding = mlp_model(data.topo[train_idx])
    combined_embedding = torch.cat((gcn_embedding, mlp_embedding), dim=1)
    mlp_emb = mlp_2(combined_embedding)
    loss = F.nll_loss(mlp_emb, data.y.squeeze()[train_idx])
    loss.backward()
    optimizer_mlp2.step()
    optimizer.step()
    optimizer_mlp.step()

    return loss.item()


def ACC(Prediction, Label):
    correct = Prediction.view(-1).eq(Label).sum().item()
    total = len(Label)
    return correct / total

def accuracy(Prediction, Label):
    correct = Prediction.view(-1).eq(Label).sum().item()
    total=len(Label)
    return correct / total
@torch.no_grad()
def test(model, mlp_model, mlp_2, data, train_idx, valid_idx, test_idx):
    model.eval()
    mlp_model.eval()
    mlp_2.eval()

    gcn_out = model(data.x, data.adj_t)
    # print(gcn_out[0])
    mlp_out = mlp_model(data.topo)
    # print(mlp_out)
    # out=torch.cat((gcn_out,mlp_out),dim=1)
    Com = torch.cat((gcn_out, mlp_out), dim=1)
    out = mlp_2(Com)
    y_pred = out.argmax(dim=-1, keepdim=True)
    # print(y_pred[0])
    y_pred = y_pred.view(-1)
    train_acc = ACC(data.y[train_idx], y_pred[train_idx])
    valid_acc = ACC(data.y[valid_idx], y_pred[valid_idx])
    test_acc = ACC(data.y[test_idx], y_pred[test_idx])
    return train_acc, valid_acc, test_acc


class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class MLP_H2GCN(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP_H2GCN, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
#             x = data.graph['node_feat']
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP_H2GCN(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        #         x = data.graph['node_feat']
        #         n = data.graph['num_nodes']
        x = data.x
        n = len(data.y)

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
