import networkit as nk
import numpy as np
import copy
import random
from torch_geometric.utils import from_networkit
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges

def rng(prob):
    '''
    Function that compute, given a probability, if it will return true or false based on a random distribuition.

    Parameters:
    prob (float): probability value to add a vertex

    Returns
    bool: True if the value is selected, False other case.
    '''
    if prob == 0:
        return False
    numero_aleatorio = random.random()
    return numero_aleatorio <= prob

def phi(d, n, alpha, beta, l, gamma = 2):
    '''
    function to compute phi (probability distribution function). If distance is zero, then returns zero. The equation is defined by:

    $\phi(u,v | P,L) = \frac{1}{{d(u,v)}^{\alpha} \cdot |P|^{\beta} \cdot L^\gamma}$

    Parameters:
    d: distance between u and v
    n: number of positive nodes
    l: number of iterations
    alpha: parameter of probability
    beta: parameter of probability
    gamma: parameter of probability

    returns:
    float: the probability of add a edge between u and v
    '''
    try:
        return 1/(d**alpha * n**beta * l ** gamma)
    except:
        return 0

def rewiring(graph, L, P, alpha = 0.2, beta = 0.5, gamma = 2):
    '''
    Function that apply the Positive and Sequential Rewiring via Breadth Search

    Parameters:
    Graph (networkit.Graph): networkit graph object to compute the searches and add the edges
    L (int): Parameter to generate graphs
    P (list): list of positive examples
    alpha, beta, gamma: parameter of probability function phi

    Returns:
    list of graphs with the sequential method applied
    '''

    # generating the list of graphs
    rewiring_graphs = list()
    rewiring_graphs.append(copy.deepcopy(graph))

    # Transform that will be used to remove duplicated edges
    transform = RemoveDuplicatedEdges() 

    for l in range(1,L):
        src = np.array([], dtype = np.int64)
        tgt = np.array([], dtype = np.int64)

        # for every element in P, compute the distance of positive vertex in P and add an edge if it's possible
        for u in P:
            dist = nk.distance.MultiTargetBFS(graph, u, P).run().getDistances()
            prob = [phi(x, len(P), alpha, beta, l, gamma) for x in dist]
            rng_result = [rng(x) for x in prob]
            tgt_aux = np.array([P[i] for i, valor in enumerate(rng_result) if valor], dtype = np.int64)
            src_aux = np.array([u] * len(tgt_aux), dtype = np.int64)
            src = np.concatenate([src, src_aux])
            tgt = np.concatenate([tgt, tgt_aux])
        graph.addEdges((src,tgt))
        rewiring_graphs.append(copy.deepcopy(graph))
    return [transform(Data(edge_index = from_networkit(graph)[0], num_nodes = graph.numberOfNodes())) for graph in rewiring_graphs]

