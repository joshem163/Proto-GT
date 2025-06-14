# Proto-Embeddings: Giving Node Features a Sense of Direction
Welcome to the repository for Pro-GNN/Pro-GT/Pro-MLP, an innovative machine learning model for Graph representation learning framework designed to enhance graph representation learning, particularly in node classification tasks. Our model utilizes a  novel approach named "Proto-Embeddings", a lightweight prototype-based transform that embeds
each node by its distances to learned semantic landmarks, grounding learning in the
attribute space’s intrinsic geometry. This implementation uses several benchmark datasets both homophilic and heterophilic including large scale dataset like OGBN-Arxiv. The code is written in Python and utilizes PyTorch and PyTorch Geometric.


# Model Architecture
The model architecture is built around a core GNN model (GraphSAGE, GCN, GAT, LINKX,), Graph transfromer based model (GPS,SGFormer) and MLP. The model pipeline involves:
- Define a *classlandmarks* for each class in the attribute space.
- Extracting *Proto-Embeddings* for each node by calculating the distance from the *Classlandmarks*.
- Use *Proto-Embeddings* for the input of baseline models.
This new *Proto-Embeddings* allow the model to learn more expressive node representations.


# Requirements
Our Pro Model depends on the followings:
Pytorch 2.4.0, Pyflagser 0.4.7, networkx 3.2.1, sklearn 1.3.0, torch_geometric 2.4.0

   
The code is implemented in python 3.11.4. 
# Datasets
In this study,  benchmark datasets have been utilized, comprising both homophilic, and heterophilic datasets, allowing the model to be evaluated across different types of graph structures. The link to access these datasets is provided below:

[cora,citeseer,pubmed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid),[computer,photo](https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Amazon.html#torch_geometric.datasets.Amazon), [coauthor](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html), [Roman-empire,Amazon-ratings, Minesweeper,Questions](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.HeterophilousGraphDataset.html), [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/)

data_utilis.py and dataset.py files  have been taken form the following link to use the similar train/test/validation spliting as the baseline model for filtered version of the Chameleon and squirell datasets. 

https://github.com/LUOyk1999/tunedGNN/tree/main/medium_graph

# Runing the  Experiments
To repeat the experiment a specific dataset, first run proto_emb.py to get the *Proto-Embedding*  after that run the train_protoGNN.py for GNN versions mplproto.py for mlp version with the following command:
- --dataset: Dataset name (options: cora, citeseer, pubmed and so on)
- --model_type: Baseline Model (GCN, GSAGE, GAT, LINKX)   

# Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed changes.
- Feel free to open issues for discussion or questions about the code.

