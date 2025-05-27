#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MLP
from tqdm import tqdm
import os

# === Seed Fixing ===
def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed to: {seed}")

# === Logger ===
class Logger:
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run].append(result)

    def print_statistics(self):
        best_results = []
        for run_results in self.results:
            best_val = max(run_results, key=lambda x: x[1])  # best val acc
            best_results.append(best_val[2])  # test acc at best val
        mean = np.mean(best_results)
        std = np.std(best_results)
        print(f"Final Test Accuracy: {mean:.4f} ± {std:.4f}")

# === Load Dataset ===
def load_dataset(name):
    assert name.lower() in ['cora', 'citeseer', 'pubmed'], "Invalid dataset name"
    dataset = Planetoid(root=f'data/{name}', name=name.capitalize(), transform=NormalizeFeatures())
    return dataset, dataset[0]

# === MLP Model ===
class MLPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.model = MLP([in_channels, hidden_channels, out_channels], dropout=dropout, norm=None)

    def forward(self, x):
        return self.model(x)

# === Train & Test ===
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x)
    preds = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = preds[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs  # train, val, test accs

# === Main Function ===
def main(dataset_name='Cora', custom_feat_file=None, runs=10, epochs=200, hidden_channels=64, lr=0.01, weight_decay=5e-4, dropout=0.5):
    fix_seed(1)
    dataset, data = load_dataset(dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # === Load Custom Features if Provided ===
    if custom_feat_file and os.path.exists(custom_feat_file):
        print(f"Loading custom node features from {custom_feat_file}")
        custom_x = torch.load(custom_feat_file, map_location='cpu')
        data.x = custom_x.to(device)
    else:
        print("Using in-built node features.")

    logger = Logger(runs)

    for run in range(runs):
        model = MLPNet(data.x.size(1), hidden_channels, dataset.num_classes, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            loss = train(model, data, optimizer)
            result = test(model, data)
            logger.add_result(run, result)

    print(f"\n===== {dataset_name.upper()} Results =====")
    logger.print_statistics()

# === Run for All Datasets ===
if __name__ == "__main__":
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    custom_feat_paths = {
        'Cora': 'CC_cora.pt',
        'CiteSeer': 'CC_citeseer.pt',
        'PubMed': 'CC_pubmed.pt'
    }

    for name in datasets:
        custom_feat_file = custom_feat_paths.get(name, None)
        main(dataset_name=name, custom_feat_file=custom_feat_file)

