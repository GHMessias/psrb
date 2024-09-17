import torch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def euclidean_distance(tensor1: torch.tensor, tensor2: torch.tensor) -> torch.tensor:
    '''
    Compute the distance between two torch tensors

    Parameters:
    tensor1: the first input tensor
    tensor2: the second input tensor

    Returns:
    The euclidean distance between tensor1 and tensor2
    '''
    return torch.sqrt(torch.sum(tensor1 - tensor2) ** 2)

def cluster_signal_ratio(cluster: list, positives: list, ratio: float = 0.5)-> int:
    '''
    Compute the signal of a cluster. If more than 0.5 of the cluster has positive labels, then return 1. Else, return 0

    Parameters:
    cluster: The cluster to determine if it's positive or negative
    positives: The list of positive elements (indexes)
    ratio: The ratio that determines if a cluster is positive or negative

    Returns:
    0 if the cluster is negative, 1 if it's positive
    '''
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > ratio * len(cluster):
        return 1
    else:
        return 0
    
def cluster_signal_abs(cluster: list, positives: list, k: int):

    '''
    Compute the signal of a cluster based in how many positive elements they have. If there is more than num_positives/k positive elements, than the cluster is positive. Else, is negative.

    Parameters:
    cluster: The cluster to determine if it's positive or negative
    positives: The list of positive elements (indexes)
    k: the number of clusters used in the MCLS algorithm

    Returns:
    0 if the cluster is negative, 1 if it's positive   
    '''
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > len(positives) / k:
        return 1
    else:
        return 0
    
def phy(m):
    '''
    Function to compute phy value of CCRNE algorithm

    Parameter: 
    m: number of positive examples

    Returns:
    torch.tensor: value of phy
    '''
    return torch.log2(m) + 1

def return_index(centroids, O_j):
    '''
    Return the index of the O_j centroid

    centroids (list): list of centroids
    O_j (torch.tensor): centroid

    Returns:
    int: index of O_j
    '''
    for index,value in enumerate(centroids):
        if torch.equal(value, O_j):
            return index
        
def mst_graph(X):
    """Returns Minimum Spanning Tree (MST) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed mst graph.
    """
    D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return csr_matrix(adj)  

def data_to_adjacency_matrix(data):
    '''
    Convert a data type (pytorch_geometric.data.data) to adjacency matrix

    Parameters:
    data: pytorch_geometric.data.data object

    Returns:
    np.array
    '''
    num_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    
    return np.array(adjacency_matrix)