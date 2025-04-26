from data_loader_het import *  # Custom data loader script to import datasets
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from logger import *  # Custom logger to track model performance across runs
from wise_emb_noProcessing import *# Wise embeddings import (custom module)
from models import objectview  # Model architectures (GCN, SAGE, GAT, MLP) import
from torch_geometric.nn import LINKX
import torch_geometric.transforms as T  # For transforming the graph data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_mask_indices(mask, run=None):
    if run is None:
        return np.where(mask)[0]
    else:
        return np.where([mask[i][run] for i in range(len(mask))])[0]


def main(args):
    # Convert args to an object-like structure for easier access
    if isinstance(args, dict):
        args = objectview(args)
    # print(args)
    dataset = load_data(args.dataset, None)
    data = dataset[0]  # First graph data object

    # Load dataset-wise embeddings using Wise embeddings based on the dataset choice
    if args.dataset in ['pubmed','wikics','roman-empire','amazon-ratings','questions']:
        wise_pca = ContextualPubmed(args.dataset)  # Load contextual embeddings for PubMed dataset
        wise = wise_embeddings_eucledian(args.dataset)  # Euclidean embeddings for PubMed
        wise_fe1 = torch.tensor(wise)  # Convert embedding to PyTorch tensor
        wise_fe2 = torch.tensor(wise_pca)
        spatial = spatial_embeddings(data)
        spatial_torch = torch.tensor(spatial)
        # Concatenate both sets of embeddings
        CC_domain = torch.cat((spatial_torch,wise_fe1, wise_fe2), 1).float()
    else:
        spatial = spatial_embeddings(data)
        spatial_torch=torch.tensor(spatial)
        wise = wise_embeddings(args.dataset)# Load wise embeddings for other datasets
        Inc_fe = torch.tensor(wise[0])
        sel_fe = torch.tensor(wise[1])
        CC_domain = torch.cat((spatial_torch,Inc_fe, sel_fe), 1).float()

    CC_domain = CC_domain.to(device)
    torch.save(CC_domain, f'CC_{args.dataset}.pt')
    # Load topological embeddings (Betti numbers) for the dataset
    topo = topological_embeddings(args.dataset)
    topo_betti0 = torch.tensor(topo[0]).float()
    topo_betti1 = torch.tensor(topo[1]).float()
    topo_fe = torch.cat((topo_betti0, topo_betti1), 1)  # Concatenate topological features


    topo_fe = topo_fe.to(device)
    ##################### save the features
    torch.save(topo_fe, f'topo_fe_{args.dataset}.pt')


if __name__ == "__main__":
    #datasets_to_run = [ 'computers','photo','physics','cs', 'wikics','pubmed']  # Add more as needed
    datasets_to_run = ['cs']
    for ds in datasets_to_run:
        print(f"\nRunning on dataset: {ds}")
        args = {
            'model_type': 'SAGE',
            'dataset': ds,
            'public_split': 'yes',
            'num_layers': 2,
            'heads': 1,
            'batch_size': 32,
            'hidden_channels': 64,
            'dropout': 0.5,
            'epochs': 400,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'runs': 10,
            'log_steps': 10,
            'weight_decay': 5e-4,
            'lr': 0.01,
            'dropout_mlp': 0.5,
        }

        main(args)

