import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch
import csv
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
def Average(lst):
    # average function
    avg = np.average(lst)
    return (avg)



def spatial_embeddings_cuda(data, test_index):
    """
    CUDA-compatible spatial embeddings.
    Produces [N, 2*C] features:
      - first C dims: 1-hop out-neighbor label counts
      - next  C dims: 2-hop out-neighbor label counts (second-hop edges exclude self-loops)

    Test nodes are relabeled to an extra class (max(y)+1), matching your code.
    """
    device = data.y.device
    y = data.y.view(-1).to(device)
    N = y.size(0)

    # Number of classes = max(label)+2 (extra class for test nodes)
    C = int(y.max().item()) + 2
    test_class = C - 1

    # test_index -> torch.LongTensor on the same device
    if not torch.is_tensor(test_index):
        test_index = torch.tensor(test_index, dtype=torch.long)
    test_index = test_index.to(device)

    # Label with test nodes mapped to extra class
    label = y.clone()
    label[test_index] = test_class

    # One-hot labels: [N, C]
    onehot = F.one_hot(label, num_classes=C).to(torch.float32)

    # Build adjacency (directed) from edge_index: row=src, col=dst
    edge_index = data.edge_index.to(device)
    adj1 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(N, N)).to(device)

    # 1-hop counts: for each i, count labels of dst where i -> dst
    hop1 = adj1.matmul(onehot)  # [N, C]

    # 2-hop counts: i -> u -> v, but ignore self-loop edges in second hop (u->u)
    mask_no_self = edge_index[0] != edge_index[1]
    edge_index2 = edge_index[:, mask_no_self]
    adj2 = SparseTensor.from_edge_index(edge_index2, sparse_sizes=(N, N)).to(device)

    hop2 = adj1.matmul(adj2.matmul(onehot))  # [N, C]

    # Final feature: [N, 2C]
    feats = torch.cat([hop1, hop2], dim=1)
    return feats


def spatial_embeddings(data,test_index):
    Node_class = list(range(max(data.y) + 2))
    n = len(data.y)
    label = data.y.clone()
    Edge_idx = data.edge_index.numpy()
    Node = range(n)
    Edge_indices = []
    for i in range(len(Edge_idx[1])):
        Edge_indices.append((Edge_idx[0][i], Edge_idx[1][i]))
    #test_index = np.where(data.test_mask)[0]
    test_class = max(data.y) + 1
    print(test_class)
    for idx_test in test_index:
        label[idx_test] = test_class

    F_vec = []
    for i in tqdm(range(n), desc="Processing spatial features"):
    #for i in range(n):
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
    # norm_fe=MinMaxScaler().fit_transform(F_vec)
    return F_vec

def spatial_one_hop(data,test_index):
    Node_class = list(range(max(data.y) + 2))
    n = len(data.y)
    label = data.y.clone()
    Edge_idx = data.edge_index.numpy()
    Node = range(n)
    Edge_indices = []
    for i in range(len(Edge_idx[1])):
        Edge_indices.append((Edge_idx[0][i], Edge_idx[1][i]))
    #test_index = np.where(data.test_mask)[0]
    test_class = max(data.y) + 1
    for idx_test in test_index:
        label[idx_test] = test_class

    F_vec = []
    for i in tqdm(range(n), desc="Processing spatial features"):
    #for i in range(n):
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


        # print(list_out)
        # print(list_In)
        for d in Node_class:
            count = 0
            count_in = 0

            for node in list_out:
                if Node_class[node] == d:
                    count += 1
            node_F.append(count)

        F_vec.append(node_F)
    # norm_fe=MinMaxScaler().fit_transform(F_vec)
    return F_vec

def Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    return intersection
# def Similarity_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Counts intersection: sum( (a & b) ) for boolean tensors.
#     Returns a scalar tensor.
#     """
#     a = a.bool()
#     b = b.bool()
#     return (a & b).sum()


def Jaccard_Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    union = np.sum(np.logical_or(array1, array2))
    jaccard_similarity = intersection / union

    return jaccard_similarity



def Russell_Rao_Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    # union = np.sum(np.logical_or(array1, array2))
    # jaccard_similarity = intersection / union

    # return jaccard_similarity
    return intersection / len(array1)



def Cosine_Similarity(array1, array2):
    # Calculate the dot product
    dot_product = np.dot(array1, array2)

    # Calculate the magnitude of each vector
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)

    # Handle zero-vector cases
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # define similarity with a zero vector as 0 (no direction)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

def Euclidean_Similarity(array1, array2):
    # Calculate Euclidean distance
    distance = np.linalg.norm(np.array(array1) - np.array(array2))

    # Convert distance to similarity (higher = more similar)
    similarity = 1 / (1 + distance)
    return similarity
def Proto_embeddings(data, dataset_name,test_idx):
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Data.head()
    label = data.y.numpy()
    if dataset_name == 'squirrel':
        Ir = 0.01
    else:
        Ir = 0.1

    Number_nodes = len(data.y)
    fe_len = len(data.x[0])

    catagories = Data['class'].to_numpy()
    Train_Data = Data.drop(index=Data.index[test_idx])
    data_by_class = {cls: Train_Data.loc[Train_Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    landmark1 = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]
    landmark2 = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * Ir))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]
    X = data.x.cpu().numpy()
    #num_cluster=len(np.unique(data.y))
    #print(num_cluster)
    #kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    #landmark3 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    Fec = []
    SFec = []
    # kFec=[]
    fe_len = len(data.x[0])
    feature_names = [ii for ii in range(fe_len)]

    #for i in tqdm(range(len(Domain_Fec)), desc="Processing contextual feature"):
    for i in range(len(Domain_Fec)):
        vec = []
        Svec = []
        # kvec=[]

        # Extract the features for the current node
        f = Domain_Fec.loc[i, feature_names].values.flatten().tolist()

        # Compute similarities for basis
        for b in landmark1:
            vec.append(Similarity(f, b))

        # Compute similarities for sel_basis
        for sb in landmark2:
            Svec.append(Similarity(f, sb))
        # for kb in landmark3:
        #     kvec.append(Cosine_Similarity(f, kb))

        # Clear the feature list and append results
        f.clear()
        Fec.append(vec)
        SFec.append(Svec)
        # kFec.append(kvec)
    # cont_fe = Contextual(data)
    # norm_Fec = MinMaxScaler().fit_transform(Fec)
    # norm_SFec = MinMaxScaler().fit_transform(SFec)
    # norm_count_fe = MinMaxScaler().fit_transform(cont_fe)
    return Fec, SFec



# def Contextual(data,test_idx):
#     # Scale data before applying PCA
#     DataAttribute = pd.DataFrame(data.x.numpy())
#     Train_Data = data.drop(index=data.index[test_idx])
#     scaling = StandardScaler()
#
#     # Use fit and transform method
#     scaling.fit(DataAttribute)
#     Scaled_data = scaling.transform(DataAttribute)
#
#     # Set the n_components=3
#     m = 100
#     principal = PCA(n_components=m)
#     principal.fit(Scaled_data)
#     x = principal.transform(Scaled_data)
#     return x

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def Contextual(data, test_idx, n_components=100, return_fitted=False):
    """
    Fit (StandardScaler + PCA) using ONLY non-test nodes, then transform ALL nodes.

    Args:
        data: PyG Data object with data.x (num_nodes, num_features)
        test_idx: indices of test nodes (list / numpy array / torch tensor)
        n_components: PCA components
        return_fitted: if True, also return (scaler, pca)

    Returns:
        x_pca: torch.FloatTensor of shape (num_nodes, n_components)
    """
    X = data.x.detach().cpu().numpy()  # (N, F)
    N = X.shape[0]

    # Normalize test_idx to a 1D numpy int array
    if torch.is_tensor(test_idx):
        test_idx = test_idx.detach().cpu().numpy()
    test_idx = np.asarray(test_idx, dtype=int).ravel()

    # Build train mask: True = use for fitting, False = test nodes
    train_mask = np.ones(N, dtype=bool)
    train_mask[test_idx] = False

    # Fit ONLY on non-test nodes
    scaler = StandardScaler()
    scaler.fit(X[train_mask])

    X_scaled_all = scaler.transform(X)  # transform ALL nodes with train-fitted scaler

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X_scaled_all[train_mask])   # fit ONLY on non-test nodes

    X_pca_all = pca.transform(X_scaled_all)  # transform ALL nodes

    x_pca = torch.from_numpy(X_pca_all).float().to(data.x.device)

    if return_fitted:
        return x_pca, scaler, pca
    return x_pca


def proto_embeddings_eucledian(data,test_idx):
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    Train_Data = Data.drop(index=Data.index[test_idx])
    data_by_class = {cls: Train_Data.loc[Train_Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    landmark1 = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    X = data.x.cpu().numpy()
    #num_cluster=len(np.unique(data.y))
    #print(num_cluster)
    #kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    #landmark2 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    Fec = []
    #kfec=[]
    #for i in range(Number_nodes):
    for i in tqdm(range(Number_nodes), desc='processing contextual features'):
        vec = []
        #kvec=[]
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(landmark1[j])))
            #kvec.append(Cosine_Similarity(np.array(f),np.array(landmark2[j])))
        f.clear()
        Fec.append(vec)
        #kfec.append(kvec)
    cont_fe=Contextual(data,test_idx)
    return Fec,cont_fe
def cosine_similarity_torch(A, B, eps=1e-12):
    """
    Vectorized cosine similarity matching your numpy version:
      cos(a,b) = dot(a,b) / (||a|| * ||b||)
      if ||a||==0 or ||b||==0 -> 0

    A: [N, F]
    B: [C, F]
    returns: [N, C]
    """
    # dot products: [N, C]
    dot = A @ B.t()

    # norms: [N, 1], [C, 1]
    An = A.norm(p=2, dim=1, keepdim=True)
    Bn = B.norm(p=2, dim=1, keepdim=True)

    denom = An * Bn.t()  # [N, C]

    # If denom == 0 -> similarity = 0 (exactly like your function)
    sim = torch.where(denom > 0, dot / denom, torch.zeros_like(dot))
    return sim

def similarity_torch(A, B, threshold=0.0):
    """
    Vectorized intersection similarity (counts of logical AND):
      sim(i,j) = sum_k [A[i,k] & B[j,k]]

    A: [N, F]
    B: [C, F]
    returns: [N, C]   (integer counts as float tensor)

    Notes:
    - If A/B are already boolean or {0,1}, set threshold=0.0 and it works.
    - If A/B are real-valued, this binarizes with (x > threshold).
    """
    A_bin = (A > threshold)
    B_bin = (B > threshold)

    # Convert to int for matrix multiply: AND count = sum(A_bin * B_bin)
    A_int = A_bin.to(torch.int32)
    B_int = B_bin.to(torch.int32)

    # [N, C] = [N, F] @ [F, C]
    sim = A_int @ B_int.t()

    # return float (often nicer downstream), but you can keep int if you want
    return sim.to(torch.float32)

def Proto_embeddings_cuda(data, dataset_name, test_idx, Ir_squirrel=0.01, Ir_other=0.1):
    """
    CUDA-compatible Proto_embeddings using cosine similarity.

    Returns:
        Fec  : [N, C] cosine similarity to landmark1 (per-class max prototype)
        SFec : [N, C] cosine similarity to landmark2 (per-class thresholded-ones prototype)
    """
    device = data.x.device
    X = data.x                      # [N, F] (GPU/CPU)
    y = data.y.view(-1).long()      # [N]
    N, F = X.shape
    C = int(y.max().item()) + 1

    Ir = Ir_squirrel if dataset_name == "squirrel" else Ir_other

    # test_idx -> torch tensor on device
    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    # train mask: exclude test nodes
    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False

    X_train = X[train_mask]
    y_train = y[train_mask]

    # landmarks
    landmark1 = torch.zeros(C, F, device=device, dtype=X.dtype)  # per-class max
    landmark2 = torch.zeros(C, F, device=device, dtype=X.dtype)  # per-class thresholded ones

    #num_cluster=len(np.unique(data.y))
    num_cluster=100
    print(num_cluster)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_train)
    landmark3 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    for cls in range(C):
        cls_mask = (y_train == cls)
        if cls_mask.any():
            Xc = X_train[cls_mask]                         # [Nc, F]
            landmark1[cls] = Xc.max(dim=0).values          # max per feature

            frac_ones = (Xc == 1).to(X.dtype).mean(dim=0)  # fraction of ones per feature
            landmark2[cls] = (frac_ones >= Ir).to(X.dtype) # 0/1 vector
        else:
            # class absent in train split -> zero prototypes (safe)
            landmark1[cls].zero_()
            landmark2[cls].zero_()

    # cosine similarities (vectorized)
    # Fec  = cosine_similarity_torch(X, landmark1)  # [N, C]
    # SFec = cosine_similarity_torch(X, landmark2)  # [N, C]
    Fec  = similarity_torch(X, landmark1)  # [N, C]
    SFec = similarity_torch(X, landmark2)  # [N, C]
    knnFec = cosine_similarity_torch(X, landmark3)  # [N, C]

    return Fec, SFec, knnFec


def proto_embeddings_euclidean_torch(data, test_idx):
    """
    CUDA-friendly replacement for proto_embeddings_eucledian.

    Returns:
      dist:  [N, C] torch.FloatTensor on same device as data.x
      cont_fe: whatever Contextual returns (see note)
    """
    device = data.x.device
    x = data.x                      # [N, F] on GPU/CPU
    y = data.y.view(-1).long()      # [N]

    N, F = x.size(0), x.size(1)
    C = int(y.max().item()) + 1

    # test_idx -> torch tensor on device
    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    # train mask excludes test nodes
    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False

    # Compute class landmarks (means) using ONLY train nodes
    landmarks = torch.zeros(C, F, device=device, dtype=x.dtype)
    counts = torch.zeros(C, device=device, dtype=x.dtype)

    x_train = x[train_mask]
    y_train = y[train_mask]

    # sum features per class
    landmarks.index_add_(0, y_train, x_train)
    counts.index_add_(0, y_train, torch.ones_like(y_train, dtype=x.dtype))

    # avoid divide-by-zero (if a class is absent in train split)
    counts = counts.clamp_min(1.0).unsqueeze(1)     # [C, 1]
    landmarks = landmarks / counts                  # [C, F]

    num_cluster=100
    print("Extracting Proto Features")
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x_train)
    landmark3 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Compute Euclidean distances from every node to every landmark:
    # dist[i,j] = ||x_i - landmark_j||
    # Use torch.cdist (GPU-accelerated)
    dist = torch.cdist(x, landmarks, p=2)           # [N, C]
    knnFec = cosine_similarity_torch(x, landmark3)

    # Contextual features:
    # If your Contextual() uses sklearn/PCA, it's CPU-only.
    # You can keep it CPU-side or rewrite it in torch.
    #cont_fe = Contextual(data,test_idx)  # replace with your version if needed

    return dist, knnFec


RESULTS_FILE = "best_results.csv"

def save_best_result(dataset_name, best_result, std, best_args, runtime):
    import csv
    import os
    import torch

    RESULTS_FILE = "best_results.csv"

    # convert tensors to Python floats
    if torch.is_tensor(best_result):
        best_result = best_result.item()
    if torch.is_tensor(std):
        std = std.item()
    if torch.is_tensor(runtime):
        runtime = runtime.item()

    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "dataset",
                "model",
                "best_test_mean",
                "best_test_std",
                "lr",
                "hidden_channels",
                "dropout",
                "num_layers",
                "runs",
                "epochs",
                "runtime_sec"
            ])

        writer.writerow([
            dataset_name,
            best_args["model_type"],
            round(best_result, 4),
            round(std, 4),
            best_args["lr"],
            best_args["hidden_channels"],
            best_args["dropout"],
            best_args["num_layers"],
            best_args["runs"],
            best_args["epochs"],
            round(runtime, 2)
        ])
