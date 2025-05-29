#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import itertools
import sys
import warnings
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MLP

warnings.filterwarnings("ignore", category=FutureWarning)

# === Seed Fixing ===
def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Logger ===
class Logger:
    def __init__(self, total_runs):
        self.results = [[] for _ in range(total_runs)]

    def add_result(self, index, result):
        self.results[index].append(result)

    def print_statistics(self):
        best_results = []
        for run_results in self.results:
            best_val = max(run_results, key=lambda x: x[1])
            best_results.append(best_val[2])
        mean = np.mean(best_results)
        std = np.std(best_results)
        return mean, std

# === Dataset Loading ===
def load_dataset(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name.capitalize(), transform=NormalizeFeatures())
    else:
        dataset = HeterophilousGraphDataset(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
    return dataset, dataset[0]

# === MLP Model ===
class MLPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.model = MLP([in_channels, hidden_channels, out_channels], dropout=dropout, norm=None)

    def forward(self, x):
        return self.model(x)

# === Train and Test ===
def train(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, train_mask, val_mask, test_mask):
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

# === Main Training ===
def main(args, use_custom_feats=True):
    dataset, data = load_dataset(args['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_custom_feats:
        cc_path = f"CC_{args['dataset_name']}.pt"
        data.x = torch.load(cc_path, map_location='cpu').to(device)
    else:
        data.x = data.x.to(device)

    data = data.to(device)

    if len(data.train_mask.shape) == 1:
        train_masks = [data.train_mask]
        val_masks = [data.val_mask]
        test_masks = [data.test_mask]
    else:
        train_masks = data.train_mask.T.bool()
        val_masks = data.val_mask.T.bool()
        test_masks = data.test_mask.T.bool()

    logger = Logger(total_runs=args['runs'] * len(train_masks))

    for split in range(len(train_masks)):
        train_mask = train_masks[split].to(device)
        val_mask = val_masks[split].to(device)
        test_mask = test_masks[split].to(device)

        for run in range(args['runs']):
            fix_seed(run)

            model = MLPNet(
                in_channels=data.x.size(1),
                hidden_channels=args['hidden_channels'],
                out_channels=dataset.num_classes,
                dropout=args['dropout']
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

            for epoch in range(1, args['epochs'] + 1):
                train(model, data, train_mask, optimizer)
                result = test(model, data, train_mask, val_mask, test_mask)
                index = split * args['runs'] + run
                logger.add_result(index, result)

    return logger.print_statistics()

# === Entry Point ===
if __name__ == "__main__":
    datasets = ['amazon-ratings', 'cora', 'citeseer']

    lr_values = [0.001]
    hidden_channels_values = [64, 128]
    dropout_values = [0.0, 0.3]

    for dataset_name in datasets:
        print(f"\n\n========== Running for dataset: {dataset_name} ==========\n")

        # === Custom Features ===
        print("\n=== Custom Feature Results ===")
        for lr, hidden_channels, dropout in itertools.product(lr_values, hidden_channels_values, dropout_values):
            args = {
                'dataset_name': dataset_name,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'epochs': 400,
                'runs': 10,
                'weight_decay': 5e-4,
                'lr': lr,
                'log_steps': 50
            }

            print(f"\nCustom Features: {args}")
            mean, std = main(args, use_custom_feats=True)
            print(f"Mean ± Std: {mean:.4f} ± {std:.4f}")

        # === Inbuilt Features ===
        print("\n=== Inbuilt Feature Results ===")
        for lr, hidden_channels, dropout in itertools.product(lr_values, hidden_channels_values, dropout_values):
            args = {
                'dataset_name': dataset_name,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'epochs': 400,
                'runs': 10,
                'weight_decay': 5e-4,
                'lr': lr,
                'log_steps': 50
            }

            print(f"\nInbuilt Features: {args}")
            mean, std = main(args, use_custom_feats=False)
            print(f"Mean ± Std: {mean:.4f} ± {std:.4f}")

