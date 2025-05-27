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
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MLP

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
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run].append(result)

    def print_statistics(self):
        best_results = []
        for run_results in self.results:
            best_val = max(run_results, key=lambda x: x[1])  # best val acc
            best_results.append(best_val[2])  # corresponding test acc
        mean = np.mean(best_results)
        std = np.std(best_results)
        print(f"\nFinal Test Accuracy: {mean:.4f} ± {std:.4f}")
        return mean, std

# === Load Dataset ===
def load_dataset(name):
    dataset = Planetoid(root=f'data/{name}', name=name, transform=NormalizeFeatures())
    return dataset, dataset[0]

# === MLP Model ===
class MLPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.model = MLP([in_channels, hidden_channels, out_channels], dropout=dropout, norm=None)

    def forward(self, x):
        return self.model(x)

# === Train & Test Functions ===
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
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs  # [train_acc, val_acc, test_acc]

# === Main Function with Args ===
def main(args):
    dataset, data = load_dataset(args['dataset_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    logger = Logger(args['runs'])

    for run in range(args['runs']):
        fix_seed(run)
        model = MLPNet(
            in_channels=dataset.num_features,
            hidden_channels=args['hidden_channels'],
            out_channels=dataset.num_classes,
            dropout=args['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        for epoch in tqdm(range(args['epochs']), desc=f'Run {run+1}/{args["runs"]} | {args["dataset_name"]}', leave=False):
            train(model, data, optimizer)
            result = test(model, data)
            logger.add_result(run, result)

    return logger.print_statistics()

# === Grid Search over Datasets ===
if __name__ == "__main__":
    dataset_names = ['Cora', 'CiteSeer', 'PubMed']
    lr_values = [0.001]
    hidden_channels_values = [64, 128]
    dropout_values = [0.0, 0.3]

    for dataset_name in dataset_names:
        print(f"\n\n==================== {dataset_name.upper()} ====================")
        best_result = 0
        best_args = None

        for lr, hidden_channels, dropout in itertools.product(lr_values, hidden_channels_values, dropout_values):
            args = {
                'dataset_name': dataset_name,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'epochs': 400,
                'runs': 10,
                'weight_decay': 5e-4,
                'lr': lr
            }

            print(f"\nTrying config: lr={lr}, hidden_channels={hidden_channels}, dropout={dropout}")
            test_acc, test_std = main(args)

            if test_acc > best_result:
                best_result = test_acc
                std = test_std
                best_args = args

        print(f"\nBest config for {dataset_name}: {best_args}")
        print(f"Best test accuracy: {best_result:.4f} ± {std:.4f}")

