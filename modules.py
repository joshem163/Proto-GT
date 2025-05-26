import networkx as nx
from networkx import ego_graph
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyflagser
from data_loader import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
def Average(lst):
    # average function
    avg = np.average(lst)
    return (avg)


def Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    return intersection


def Degree_list(Graph):
    degree_list = [Graph.degree(node) for node in Graph.nodes]
    return np.array(degree_list)


def wise_embeddings(data):

    # Convert node features and labels to pandas DataFrame
    Domain_Fec = pd.DataFrame(data.x.numpy())  # Feature matrix as a DataFrame
    label = pd.DataFrame(data.y.numpy(), columns=['class'])  # Labels as a DataFrame
    Data = pd.concat([Domain_Fec, label], axis=1)  # Combine features and labels

    # Get dataset statistics
    Number_nodes = len(data.y)  # Number of nodes
    fe_len = len(data.x[0])  # Feature vector length
    catagories = Data['class'].to_numpy()  # Convert labels to a NumPy array

    # Group data by class and remove the class column
    data_by_class = {
        cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1)
        for cls in range(max(catagories) + 1)
    }

    # Compute basis vectors (maximum value of each feature per class)
    basis = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]

    # Compute selected basis (binary selection based on frequency threshold)
    sel_basis = [
        [int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * 0.1))
         for i in range(len(df.columns))]
        for df in data_by_class.values()
    ]

    # Generate feature indices
    feature_names = [ii for ii in range(fe_len)]

    # Initialize lists to store computed feature embeddings
    Fec = []  # Stores similarity with basis vectors
    SFec = []  # Stores similarity with selected basis vectors

    for i in range(Number_nodes):
        vec = []  # Stores similarity scores for basis
        Svec = []  # Stores similarity scores for selected basis

        # Extract the feature vector of the current node
        f = Data.loc[i, feature_names].values.flatten().tolist()

        # Compute similarity of the current node's features with each basis vector
        for b in basis:
            vec.append(Similarity(f, b))

        # Compute similarity of the current node's features with each selected basis vector
        for sb in sel_basis:
            Svec.append(Similarity(f, sb))

        # Clear feature vector (not necessary here)
        f.clear()

        # Append computed similarities to the respective lists
        Fec.append(vec)
        SFec.append(Svec)

    # Return the computed Wise embeddings
    return Fec, SFec

def Contextual(data):
    # dataset = load_data(dataset_Name, None)
    # data = dataset[0]
    Domain_Fec = pd.DataFrame(data.x.numpy())

    # Scale data before applying PCA
    scaling = StandardScaler()

    # Use fit and transform method
    scaling.fit(Domain_Fec)
    Scaled_data = scaling.transform(Domain_Fec)

    # Set the n_components=3
    m = 100
    principal = PCA(n_components=m)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)
    return x

def wise_embeddings_eucledian(data):
    print("Extracting contextual features....... ")
    # dataset = load_data(dataset_Name, None)
    # data = dataset[0]
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    sel_basis = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    Fec = []
    for i in range(Number_nodes):
        #print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        vec = []
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(sel_basis[j])))
        f.clear()
        Fec.append(vec)
    return Fec


def spatial_embeddings(data):
    Node_class = list(range(max(data.y) + 1))
    print(data)
    n = len(data.y)
    label = data.y.clone()
    Edge_idx = data.edge_index.numpy()
    Node = range(n)
    Edge_indices = []
    for i in range(len(Edge_idx[1])):
        Edge_indices.append((Edge_idx[0][i], Edge_idx[1][i]))

    F_vec = []
    for i in range(n):
        print("\rProcessing file {} ({}%)".format(i, 100*i//(n-1)), end='', flush=True)
        node_F = []
        list_out = []
        list_In = []
        S_nbd_out = []
        S_nbd_in = []
        for edge in Edge_indices:
            src, dst = edge
            if src == i:
                list_out.append(label[dst])
                for edge_2 in Edge_indices:
                    src_2, dst_2 = edge_2
                    if src_2 == dst and src_2 != dst_2:
                        S_nbd_out.append(label[dst_2])

        # print(list_out)
        # print(list_In)
        for d in Node_class:
            count = 0
            count_in = 0

            for node in list_out:
                if Node_class[node] == d:
                    count += 1
            node_F.append(count)

        for d in Node_class:
            count_S_out = 0
            count_S_in = 0
            for node in S_nbd_out:
                if Node_class[node] == d:
                    count_S_out += 1
            node_F.append(count_S_out)

        F_vec.append(node_F)
    return F_vec
#
