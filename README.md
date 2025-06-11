# Proto-Embeddings: Giving Node Features a Sense of Direction
Welcome to the repository for Pro-GNN/Pro-GT/Pro-MLP, an innovative machine learning model for Graph representation learning framework designed to enhance graph representation learning, particularly in node classification tasks. Our model utilizes a  novel approach named "Proto-Embeddings", a lightweight prototype-based transform that embeds
each node by its distances to learned semantic landmarks, grounding learning in the
attribute space’s intrinsic geometry. This implementation uses several benchmark datasets both homophilic and heterophilic including large scale dataset like OGBN-Arxiv. The code is written in Python and utilizes PyTorch and PyTorch Geometric.

![FrameWork-1](https://github.com/joshem163/WISE-GNN/assets/133717791/89269231-6105-4529-bdb1-9cbc59695eb3)

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

[computer,photo]https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Amazon.html#torch_geometric.datasets.Amazon

# Runing the  Experiments
To repeat the experiment a specific dataset, run the train_*.py file with the following command:
- --dataset: Dataset name (options: cora, citeseer, pubmed, texas, cornell, wisconsin, chameleon)
- --model_type: Baseline Model (GCN, GSAGE, GAT, LINKX, H2GCN)
- --public_split: yes/no (yes: cora, citeseer, and pubmed)   

# Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed changes.
- Feel free to open issues for discussion or questions about the code.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

