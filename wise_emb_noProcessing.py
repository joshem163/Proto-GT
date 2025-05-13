import networkx as nx
from networkx import ego_graph
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyflagser
from data_loader_het import *
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

# def wise_embeddings(dataset_Name):
#     print("Extracting contextual features....... ")
#     # Load the dataset based on the given name
#     dataset = load_data(dataset_Name, None)
#     data = dataset[0]  # Assuming dataset is a list, selecting the first element
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def ContextualPubmed(dataset_Name):
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
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

def wise_embeddings_eucledian(dataset_Name):
    print("Extracting contextual features....... ")
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
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


def Topological_Feature_subLevel(adj, filtration_fun, Filtration):
    betti_0 = []
    betti_1 = []
    for p in range(len(Filtration)):
        n_active = np.where(np.array(filtration_fun) <= Filtration[p])[0].tolist()
        Active_node = np.unique(n_active)
        if (len(Active_node) == 0):
            betti_0.append(0)
            betti_1.append(0)
        else:
            b = adj[Active_node, :][:, Active_node]
            my_flag = pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2,
                                                   approximation=None)
            x = my_flag["betti"]
            betti_0.append(x[0])
            betti_1.append(x[1])
        n_active.clear()
    return betti_0, betti_1


def compute_hks(graph, t):
    """
    Compute the Heat Kernel Signature (HKS) for each node in the graph.
    :param graph: NetworkX graph (undirected, unweighted)
    :param t_values: List of diffusion time values to compute HKS
    :return: Dictionary with nodes as keys and HKS values as lists
    """
    # Compute the combinatorial Laplacian L = D - A
    L = nx.laplacian_matrix(graph).toarray()

    # Compute eigenvalues and eigenvectors of L
    eigvals, eigvecs = np.linalg.eigh(L)  # Since L is symmetric

    # Compute HKS for each node at different t values
    hks = {node: [] for node in graph.nodes()}
    heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t * eigvals)), eigvecs.T))
    for i, node in enumerate(graph.nodes()):
        hks[node].append(heat_kernel[i, i])  # Diagonal element for self-heat diffusion

    #     for t in t_values:
    #         heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t * eigvals)), eigvecs.T))
    #         for i, node in enumerate(graph.nodes()):
    #             hks[node].append(heat_kernel[i, i])  # Diagonal element for self-heat diffusion

    return hks

def topological_embeddings(dataset_Name):
    print("Extracting topological features....... ")
    dataset = load_data(dataset_Name, None)
    data = dataset[0]
    #print(data)
    Number_nodes = len(data.y)
    Edge_idx = data.edge_index.numpy()
    Node = range(Number_nodes)
    Edgelist = []
    for i in range(len(Edge_idx[1])):
        Edgelist.append((Edge_idx[0][i], Edge_idx[1][i]))
    # print(Edgelist)
    # a "plain" graph is undirected
    G = nx.Graph()

    # give each a node a 'name', which is a letter in this case.
    # G.add_node('a')

    # the add_nodes_from method allows adding nodes from a sequence, in this case a list
    # nodes_to_add = ['b', 'c', 'd']
    G.add_nodes_from(Node)

    # add edge from 'a' to 'b'
    # since this graph is undirected, the order doesn't matter here
    # G.add_edge('a', 'b')

    # just like add_nodes_from, we can add edges from a sequence
    # edges should be specified as 2-tuples
    # edges_to_add = [('a', 'c'), ('b', 'c'), ('c', 'd')]
    G.add_edges_from(Edgelist)
    #node_hks = compute_hks(G, 0.1)
    #flat_hks = [node_hks[node][0] for node in sorted(node_hks)]
    flat_degree=[G.degree(node) for node in G.nodes]
    q_points = np.linspace(0, 1, 10)# quantiles=10
    thresholds = np.quantile(flat_degree, q_points)
    #print(thresholds)
    # Assign HKS as a node attribute
    for idx, deg_val in enumerate(flat_degree):
        G.nodes[idx]['deg'] = deg_val

    topo_betti_0 = []
    topo_betti_1 = []
    Node_Edge = []
    for i in range(Number_nodes):
        #print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        subgraph = ego_graph(G, i, radius=2, center=True, undirected=True, distance=None)

        # Use HKS as filtration instead of degree
        filt = [subgraph.nodes[n]['deg'] for n in subgraph.nodes]

        A_sub = nx.to_numpy_array(subgraph)  # adjacency matrix of subgraph
        fe = Topological_Feature_subLevel(A_sub, filt, thresholds)

        topo_betti_0.append(fe[0])
        topo_betti_1.append(fe[1])
        Node_Edge.append([subgraph.number_of_nodes(), subgraph.number_of_edges()])

    return topo_betti_0, topo_betti_1


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
        # print("\rProcessing file {} ({}%)".format(i, 100*i//(n-1)), end='', flush=True)
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
#
# def spatial_embeddings(data):
#     Node_class = list(range(max(data.y) + 2))
#     print(data)
#     n = len(data.y)
#     label = data.y.clone()
#     Edge_idx = data.edge_index.numpy()
#     Node = range(n)
#     Edge_indices = []
#     for i in range(len(Edge_idx[1])):
#         Edge_indices.append((Edge_idx[0][i], Edge_idx[1][i]))
#     test_index = np.where(data.test_mask)[0]
#     #idx_test = [data.test_mask[i][0] for i in range(len(data.y))]
#     #test_index = np.where(idx_test)[0]
#     test_class = max(data.y) + 1
#     for idx_test in test_index:
#         label[idx_test] = test_class
#
#     F_vec = []
#     for i in tqdm(range(n), desc="Processing spatial features"):
#         # print("\rProcessing file {} ({}%)".format(i, 100*i//(n-1)), end='', flush=True)
#         node_F = []
#         list_out = []
#         list_In = []
#         S_nbd_out = []
#         S_nbd_in = []
#         for edge in Edge_indices:
#             src, dst = edge
#             if src == i:
#                 list_out.append(label[dst])
#                 for edge_2 in Edge_indices:
#                     src_2, dst_2 = edge_2
#                     if src_2 == dst and src_2 != dst_2:
#                         S_nbd_out.append(label[dst_2])
#
#         # print(list_out)
#         # print(list_In)
#         for d in Node_class:
#             count = 0
#             count_in = 0
#
#             for node in list_out:
#                 if Node_class[node] == d:
#                     count += 1
#             node_F.append(count)
#
#         for d in Node_class:
#             count_S_out = 0
#             count_S_in = 0
#             for node in S_nbd_out:
#                 if Node_class[node] == d:
#                     count_S_out += 1
#             node_F.append(count_S_out)
#
#         F_vec.append(node_F)
#     return F_vec

def wise_embeddings_eucledian_mag( Domain_Fe,Label):
    Domain_Fec = pd.DataFrame(Domain_Fe.numpy())
    label = pd.DataFrame(Label.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(label)
    print(Number_nodes)
    fe_len = len(Domain_Fe[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    sel_basis = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    Fec = []
    for i in range(Number_nodes):
        print("\rProcessing file {} ({}%)".format(i, 100 * i // (Number_nodes - 1)), end='', flush=True)
        vec = []
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(sel_basis[j])))
        f.clear()
        Fec.append(vec)
    return Fec

def l1_normalize_row_in_parts(X, num_parts=4):
    B, D = X.shape
    assert D % num_parts == 0, "Feature dimension must be divisible by number of parts."
    part_size = D // num_parts

    X_normalized = X.clone()

    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size
        part = X[:, start:end]

        part_sum = part.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
        part_norm = part / part_sum

        X_normalized[:, start:end] = part_norm

    return X_normalized