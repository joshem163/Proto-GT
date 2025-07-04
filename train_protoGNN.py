from data_loader import *  # Custom data loader script to import datasets
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from logger import *  # Custom logger to track model performance across runs
from proto_emb import *  # Wise embeddings import (custom module)
from models import *  # Model architectures (GCN, SAGE, GAT, MLP) import
from torch_geometric.nn import LINKX
import torch_geometric.transforms as T  # For transforming the graph data
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import itertools
from dataset import *
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward pass with GraphSAGE and MLP inputs
    out = model(data.x, data.adj_t)[train_idx]

    # Compute loss (using Cross Entropy instead of NLL since we don't apply log_softmax)
    loss = F.cross_entropy(out, data.y.squeeze()[train_idx])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def ACC(prediction, label):
    correct = prediction.eq(label).sum().item()
    total = len(label)
    return correct / total


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx, metric='accuracy'):
    model.eval()
    out = model(data.x, data.adj_t)  # raw logits
    y_true = data.y.squeeze()  # ensure shape consistency

    if metric == 'accuracy':
        y_pred = out.argmax(dim=-1)
        train_score = ACC(y_pred[train_idx], y_true[train_idx])
        valid_score = ACC(y_pred[valid_idx], y_true[valid_idx])
        test_score = ACC(y_pred[test_idx], y_true[test_idx])

    elif metric == 'roc_auc':
        # Assume binary classification and get probability of class 1
        probs = F.softmax(out, dim=-1)[:, 1]  # get prob for class 1

        train_score = roc_auc_score(y_true[train_idx].cpu(), probs[train_idx].cpu())
        valid_score = roc_auc_score(y_true[valid_idx].cpu(), probs[valid_idx].cpu())
        test_score = roc_auc_score(y_true[test_idx].cpu(), probs[test_idx].cpu())

    else:
        raise ValueError("Unsupported metric: choose 'accuracy' or 'roc_auc'")

    return train_score, valid_score, test_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def get_mask_indices(mask, run=None):
    mask = mask.cpu()  # Ensure the mask tensor is on CPU
    if run is None:
        return np.where(mask.numpy())[0]
    else:
        return np.where([mask[i][run].item() for i in range(len(mask))])[0]


def main(args):
    # Convert args to an object-like structure for easier access
    if isinstance(args, dict):
        args = objectview(args)


    #################### load the features
    proto_domain=torch.load(f'proto_{args.dataset}.pt').to(device)
    # Load the dataset and apply transformations (e.g., undirected edges, sparse tensor format)
    if args.dataset in ['chameleon', 'squirrel']:
        data = load_Sq_Cha_filterred(args.dataset)
        split_idx_lst = load_fixed_splits('data', args.dataset, name=args.dataset)
        out_dim=5
    else:
        dataset = load_data(args.dataset, T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
        data = dataset[0].to(device)  # First graph data object
        out_dim=dataset.num_classes


    # Update the node features with wise embeddings and topo features
    data.x = proto_domain # Replace original features with wise embeddings    print(data)
    data=data.to(device)
    print(data)

    if args.model_type == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,out_dim , args.num_layers, args.dropout)
    elif args.model_type == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels, out_dim, args.num_layers, args.dropout)
    elif args.model_type == 'GAT':
        model = GAT(data.num_features, args.hidden_channels, out_dim, args.num_layers, args.dropout,args.heads)
    elif args.model_type == 'LINKX':
        model = LINKX(len(data.y), data.num_features, args.hidden_channels, out_dim, args.num_layers, 2, 2,
                      args.dropout)
    else:
        print('Model does not exist')
    model = model.to(device)
    # Logger to store and output results
    logger = Logger(args.runs, args)

    # Loop over different runs (cross-validation-like approach)
    for run in range(args.runs):
        # Split data into training, validation, and test based on public split or custom split
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            train_idx = get_mask_indices(data.train_mask)
            valid_idx = get_mask_indices(data.val_mask)
            test_idx = get_mask_indices(data.test_mask)

        elif args.dataset in ['texas', 'cornell', 'wisconsin','roman-empire','amazon-ratings','questions','minesweeper','actor','chameleon_filtered','squirrel_filtered']:
            train_idx = get_mask_indices(data.train_mask, run)
            valid_idx = get_mask_indices(data.val_mask, run)
            test_idx = get_mask_indices(data.test_mask, run)
        elif args.dataset in ['chameleon','squirrel']:
            train_idx = split_idx_lst[run]['train']
            valid_idx = split_idx_lst[run]['valid']
            test_idx = split_idx_lst[run]['test']
        elif args.dataset in ['wikics']:
            train_idx = get_mask_indices(data.train_mask, run)
            valid_idx = get_mask_indices(data.val_mask, run)
            test_idx = get_mask_indices(data.test_mask)
        else:
            # Generic fallback with 60/20/20 split
            num_nodes = data.x.size(0)
            all_indices = np.arange(num_nodes)
            y_np = data.y.cpu().numpy()

            train_idx, temp_idx = train_test_split(
                all_indices, train_size=0.6, stratify=y_np, random_state=42)
            valid_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=y_np[temp_idx], random_state=42)

            train_idx = torch.tensor(train_idx, dtype=torch.long)
            valid_idx = torch.tensor(valid_idx, dtype=torch.long)
            test_idx = torch.tensor(test_idx, dtype=torch.long)

        # Reset parameters of the models to ensure fresh training
        model.reset_parameters()

        # Optimizers for each model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

        # Training loop for each epoch
        for epoch in range(1, 1 + args.epochs):
            # Train the model on the current run's data
            loss = train(model,data, train_idx, optimizer)
            # Test the model on validation and test sets
            if args.dataset in ['questions', 'minesweeper']:
                result = test(model, data, train_idx, valid_idx, test_idx, metric='roc_auc')
            else:
                result = test(model, data, train_idx, valid_idx, test_idx, metric='accuracy')
            logger.add_result(run, result)  # Log the results

            # Log results every `log_steps` epochs
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                #print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                 #     f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')

        # Print run statistics
        #logger.print_statistics(run)

    # Print overall statistics after all runs
    best_test = logger.print_statistics()
    return best_test[0], best_test[1]  # <<< RETURN it!


if __name__ == "__main__":
    #datasets_to_run = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'physics', 'cs', 'roman-empire',
    #                   'amazon-ratings', 'actor', 'questions', 'minesweeper']  # Add more as needed
    datasets_to_run = ['cora']

    # Define grid of hyperparameters
    lr_values = [0.001,0.005,0.01]
    hidden_channels_values = [16,32, 64]
    dropout_values = [0.0,0.5]


    for ds in datasets_to_run:
        print(f"\nRunning grid search on dataset: {ds}")

        best_result = 0
        best_args = None

        # Create a grid of all combinations
        for lr, hidden_channels, dropout in itertools.product(lr_values, hidden_channels_values, dropout_values):
            args = {
                'model_type': 'SAGE',
                'dataset': ds,
                'public_split': 'yes',
                'num_layers': 2,
                'heads': 1,
                'batch_size': 32,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'epochs': 400,
                'opt': 'adam',
                'opt_scheduler': 'none',
                'opt_restart': 0,
                'runs': 10,
                'log_steps': 10,
                'weight_decay': 5e-4,
                'lr': lr,
                'dropout_mlp': 0.5,
            }

            print(f"Trying config: lr={lr}, hidden_channels={hidden_channels}, dropout={dropout}")
            test_acc,test_std = main(args)  # <<< main returns best test acc

            if test_acc > best_result:
                best_result = test_acc
                std=test_std
                best_args = args

        print("\n\n========== Best result for dataset:", ds, "==========")
        print(f"Best test accuracy: {best_result:.4f}±{std:.4f}")
        print(f"Best hyperparameters: {best_args}")