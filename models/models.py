from models.auxiliar_functions import *
from sklearn.cluster import KMeans
import numpy as np
from networkit.nxadapter import nk2nx
from torch_geometric.utils import to_networkit
import networkit as nk
import networkx as nx
import copy
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn
from sklearn.svm import OneClassSVM


class MCLS:
    '''
    Class that implements the MCLS algorithm.

    Main changes: Considering there is no waranty that if it's possible to correct classify a cluster (the algorithm can classify all clusters as negative), we change the way of compute if a cluster is negative or positive. If the algorithm classify all clusters as negatives and there is more than num_positive/k elements positives in a cluster, then, this cluster is positive.

    This change guarantees that the algorithm can be run.
    '''
    def __init__(self, data, k = 7, ratio = 0.3):
        self.positives = data.P
        self.k = k
        self.ratio = ratio
        self.data = data.x
        self.distance = dict()
   
    def train(self):
        '''
        Method that trains the model
        '''

        # Start the kmeans algorithm
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.data.detach().numpy())
        clusters_labels = kmeans.labels_

        # Dictionary to fill with the clusters
        clusters = {}

        # filling the clusters
        for indice, rotulo in enumerate(clusters_labels):
            if rotulo not in clusters:
                clusters[rotulo] = []
            clusters[rotulo].append(indice)

        # dictionary to fill with cluster labels
        cluster_signals = {}

        # labeling clusters. If there is no clusters with ratio = 0.5, the algorithm uses the num_positives/k value to determine if a cluster is positive
        for cluster in clusters:
            sig = cluster_signal_ratio(clusters[cluster], self.positives,ratio = self.ratio)
            cluster_signals[cluster] = sig
        if np.sum(list(cluster_signals.values())) == 0:
            cluster_signals = {}
            for cluster in clusters:
                sig = cluster_signal_abs(clusters[cluster], self.positives, self.k)
                cluster_signals[cluster] = sig

        # Saving centroids
        cluster_centroids = {}
        centroids = kmeans.cluster_centers_

        for i, center in enumerate(centroids):
            cluster_centroids[i] = center
        
        # list to save the elements of positive and negative clusters
        positive_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 1]
        negative_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 0]

        # computing positive centroids
        positive_centroids = torch.tensor(np.array([cluster_centroids[i] for i in positive_clusters]))

        # filling the distance dictionary with the most distant elements
        for cluster in negative_clusters:
            for element in clusters[cluster]:
                distances = [euclidean_distance(self.data[element], centroid) for centroid in positive_centroids]
                mean_distance = torch.mean(torch.stack(distances))
                self.distance[element] = mean_distance

    def negative_inference(self, num_neg):
        '''
        Method to sort the elements from farest to nearest to return the most distant

        Parameters:
        num_neg: number of elements to classify as reliable negatives

        Returns:
        list: list with num_neg elements classify as reliable negatives
        '''
        RN = sorted(self.distance, key=self.distance.get, reverse=True)
        RN = RN[:num_neg]
        return RN
    
class CCRNE:
    '''
    Class that implement CCRNE algorithm

    Main changes: To guarantee that all the elements will no be considered as negatives, we add the parameter ratio. This parameter is responsible for reducing the length of r_p.
    '''
    def __init__(self, data, ratio = 0.3):
        self.clusters = dict()
        self.data = data.x
        self.r_p = 0
        self.positives = data.P
        self.unlabeled = data.U
        self.ratio = ratio

    def train(self):
        '''
        Method to train the model
        '''

        # Crating a mask for positive elements
        self.pul_mask = torch.zeros(len(self.data))
        for i in self.positives:
            self.pul_mask[i] = 1
        self.pul_mask = self.pul_mask.bool()

        # Setting the centroid O_p and the value r
        O_p = self.data[self.pul_mask].mean(dim = 0)
        r = torch.max(torch.tensor([euclidean_distance(x_k, O_p) for x_k in self.data[self.pul_mask]]))

        # Computing r_p and phy
        m = torch.tensor(len(self.positives))
        self.r_p = (r * phy(m))/(phy(m) + 1)


        # Associating every cluster to a centroid
        self.clusters[0] = {'cluster' : [self.positives[0]],
                    'centroid' : self.data[self.positives[0]]}

        # Clustering element given r_p
        Z = self.positives[1:]
        n = 1

        for x_i in Z:
            lista_distancia = torch.tensor([euclidean_distance(self.data[x_i], O_k) for O_k in [self.clusters[i]['centroid'] for i in range(len(self.clusters))]])
            centroids = torch.tensor([self.clusters[i]['centroid'].tolist() for i in range(len(self.clusters))])
            order_idnex = torch.argsort(lista_distancia)
            centroids_ordenado = centroids[order_idnex]

            O_j = centroids_ordenado[0]
            j = return_index(centroids, O_j)

            if euclidean_distance(self.data[x_i], O_j) < self.r_p:
                self.clusters[j]['cluster'].append(x_i)
                O_j = (n * O_j + self.data[x_i]) / (n + 1)
                n += 1
                self.clusters[j]['centroid'] = O_j
            
            else:
                self.clusters[(len(self.clusters))] = dict()
                self.clusters[len(self.clusters) - 1]['cluster'] = [x_i]
                self.clusters[len(self.clusters) - 1]['centroid'] = self.data[x_i]

    
    def negative_inference(self, num_neg):
        '''
        Function to compute the reliable negatives

        Parameters:
        num_neg (int): Number of elements to return as reliable negatives

        Returns:
        list: List of reliable negatives indexes
        '''
        RN = self.unlabeled

        for i in range(len(self.clusters)):
            for x_i in RN:
                    if euclidean_distance(self.data[x_i], self.clusters[i]['centroid']) < self.ratio * self.r_p:
                        RN = RN[RN != x_i]
        return RN[:num_neg]

class LP_PUL:
    '''
    Class that implement LP_PUL algorithm.

    Main changes: If the graph is not connected, then it's impossible to compute the breadth first search length of every node to every node. To avoid this, we add the following criteria: If the graph is not connected then we create the Minimum Spanning Tree (MST) if the graph and add all the vertex that are not previously on the graph.
    '''
    def __init__(self, data):
        self.graph = to_networkit(data.edge_index, directed=False)
        self.data = data.x
        self.positives = data.P
        self.unlabeled = data.U

        
    def train(self):
        '''
        Method to train the model
        '''

        # Verify if it's conneted. If not, connetc the graph with the MST
        is_connected = nk.components.ConnectedComponents(self.graph).run().numberOfComponents() == 1
        if not is_connected:
            aux_g = nk2nx(self.graph)
            adj = nx.to_scipy_sparse_array(aux_g)  # Convert to sparse matrix
            adj_aux = mst_graph(self.data).toarray()
            
            rows, cols = np.where((adj.toarray() == 0) & (adj_aux == 1))
            
            for i, j in zip(rows, cols):
                self.graph.addEdge(i, j)

        # Distance vector
        self.a = np.zeros(len(self.unlabeled))

        # Compute the minimum paths with MultiTargetBFS function
        for p in self.positives:
            d = nk.distance.MultiTargetBFS(self.graph, p, self.unlabeled).run().getDistances()
            self.a += d
        self.a = self.a / len(self.positives)

    def negative_inference(self, num_neg):
        '''
        Method to classify the reliable negative examples (vertex)

        Parameter:
        num_neg (int): number of reliable negatives to returns

        Return:
        list: list of reliable negatives (indexes)
        '''
        RN = torch.stack([x for _, x in sorted(zip(self.a, self.unlabeled), reverse=True)][:num_neg])
        return RN
    
class PU_LP:
    '''
    Class to implement the PU_LP algorithm
    '''
    def __init__(self, data, alpha=0.1, m = 3, l = 1):
        self.positives = data.P 
        self.data_graph = data_to_adjacency_matrix(data)
        self.unlabeled = data.U
        self.alpha = alpha
        self.m = m
        self.l = l
        
    def train(self):
        '''
        Method to train the model

        Main changes: Considering the fact that the model automatically returns the number of reliable negative elements, we change the number of elements to be returned. This action is justified by the fact that the benchmark needs to be more fair possible.
        '''

        # Setting the dinamic parameters
        self.RP = torch.tensor([])
        P_line = copy.copy(self.positives)
        U_line = copy.copy(self.unlabeled)

        # Computing W = (I - alpha*A)**-1 - I
        I = np.eye(len(self.data_graph))
        W = np.linalg.inv(I - self.alpha * self.data_graph) - I
        W = torch.tensor(W)

        # Ranking the most similar elements based on the selected positives
        for k in range(self.m):
            rank_dict = dict()
            for vi in U_line:
                S_vi = 0
                for vj in P_line:
                    S_vi += W[vi, vj]
                S_vi /= len(P_line)
                rank_dict[vi] = S_vi
        rank_dict = sorted(rank_dict.items(), key=lambda x:x[1], reverse=True)
        rank_dict = [tupla[0] for tupla in rank_dict]

        # Updating sets
        RP_line = torch.tensor(rank_dict[:int((self.l / self.m) * len(self.positives))])
        P_line = torch.cat([P_line,RP_line], dim = 0)
        U_line = torch.tensor(list(set(U_line) - set(RP_line)))
        self.RP = torch.cat([self.RP,RP_line], dim = 0)

        # Classifying the reliable negatives
        rank_dict = dict()
        for vi in list(set(self.unlabeled) - set(RP_line)):
            S_vi = 0
            for vj in torch.cat([self.positives, self.RP], dim = 0):
                
                S_vi += W[vi, int(vj)]
            S_vi /= len(P_line)
            rank_dict[vi] = S_vi
        rank_dict = sorted(rank_dict.items(), key=lambda x:x[1])
        rank_dict = [tupla[0] for tupla in rank_dict]
        self.RN = rank_dict[:len(torch.cat([self.positives,self.RP], dim = 0))]

    def negative_inference(self, num_neg = None):
        '''
        Method that returns the reliable negative examples. The num_neg value can be set to none and the algorithm will return the number of negative elements based on the proposed method.

        Parameters:
        num_neg (int): Number of reliable negative elements to return

        Returns:
        torch.tensor: tensor of negative elements.
        '''
        #return self.RN[-len(self.positives + self.RP):][:num_neg]
        if not num_neg:
            num_neg = len(self.RN) + len(self.RP)
        return torch.stack(self.RN[-num_neg:])
    
    
    def positive_inference(self):
        '''
        Method that returns the positive elements (computed at the training phase)
        '''
        return torch.stack(self.RP)
    
class RCSVM:
    '''
    Class to implement the RCSVM algorithm
    
    Main changes: Considering the fact that the model automatically returns the number of reliable negative elements, we change the number of elements to be returned. This action is justified by the fact that the benchmark needs to be more fair possible.
    '''
    def __init__(self, data, alpha = 0.7, beta = 0.3):
        self.data = data.x
        self.positives = data.P
        self.unlabeled = data.U 
        self.alpha = alpha
        self.beta = beta

    def train(self):
        '''
        Method to train the model
        '''

        # sum of positive elements
        soma_positive = 0
        for element in self.positives:
            soma_positive += self.data[element] / torch.norm(self.data[element], p = 2)
            
        # sum of unlabeled elements
        soma_unlabeled = 0
        for element in self.unlabeled:
            soma_unlabeled += self.data[element] / torch.norm(self.data[element], p = 2)


        # representant vectors
        self.c_positive = (self.alpha / len(self.positives)) * soma_positive - (self.beta / len(self.unlabeled)) * soma_unlabeled 
        self.c_negative = (self.alpha/ len(self.unlabeled)) *  soma_unlabeled - (self.beta / len(self.positives)) * soma_positive

    def negative_inference(self, num_neg = None, similarity = torch.dist):
        '''
        Method that will return the reliable negative examples. The num_neg value can be set to none and the algorithm will return the number of negative elements based on the proposed method.

        Parameters:
        num_neg (int): Number of reliable negative elements to return
        similarity: similarity to be used 

        Returns:
        torch.tensor: tensor of negative elements.       
        '''
        RN = list()
        for element in self.unlabeled:
            if similarity(self.c_positive, self.data[element]) <= similarity(self.c_negative,self.data[element]):
                RN.append(element)
        
        if len(RN) == 0:
            return []
        if not num_neg:
            return torch.tensor(RN)
        else:
            return torch.tensor(RN[:num_neg])
        
class RGCN_layer(torch.nn.Module):
    '''
    Class to implement the Relational Graph Convolutional Network (RGCN). This Relational module is used for every graph generated via rewiring
    '''
    def __init__(self, input_size, output_size, L):
        super(RGCN_layer, self).__init__()
        self.L = L
        self.input_size = input_size
        self.output_size = output_size
        self.parameter_list = torch.nn.ParameterList()

        # Rewiring layer (relational layer)
        self.rewiring_sublayers = list()
        for i in range(self.L):
            sublayer = GCNConv(self.input_size, self.output_size)
            self.rewiring_sublayers.append(sublayer)
            for param in self.rewiring_sublayers[i].parameters():
                self.parameter_list.append(param)

    def forward(self, x, rewiring_graph_list):
        # Computing the sum for every generated graph
        out = torch.zeros((x.shape[0], self.output_size))
        for i in range(self.L):
            _x = self.rewiring_sublayers[i](x, rewiring_graph_list[i].edge_index)
            out += _x
        return out
    
class RGCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, L, output_activation_function = torch.relu):
        super(RGCN, self).__init__()
        self.layer1 = RGCN_layer(input_size, hidden_size, L)
        self.layer2 = RGCN_layer(hidden_size, output_size, L)
        self.output_activation_function = output_activation_function
        self.L = L
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x, rewiring_graph_list):
        x = self.layer1(x, rewiring_graph_list)
        x = torch.relu(x)
        x = self.layer2(x, rewiring_graph_list)
        return x
    
class ExtendedRGCN(torch.nn.Module):
    def __init__(self, base_model, add_layer_dim, output_activation_function = torch.softmax):
        super(ExtendedRGCN, self).__init__()
        self.base_model = base_model
        # self.added_layer = RGCN_layer(base_model.output_size, add_layer_dim, base_model.L)
        # self.added_layer = GCNConv(base_model.output_size, add_layer_dim)
        self.added_layer = torch.nn.Linear(base_model.output_size, add_layer_dim)
        self.output_activation_function = output_activation_function

    def forward(self, x, rewiring_graph_list):
        x = self.base_model(x, rewiring_graph_list)
        # x = self.output_activation_function(self.added_layer(x, rewiring_graph_list), dim = 1)
        # x = self.output_activation_function(self.added_layer(x, rewiring_graph_list[0].edge_index), dim = 1)
        x = self.output_activation_function(self.added_layer(x), dim = 1)

        return x
    
class GCN(torch.nn.Module):
    '''
    Class to implement the Graph Convolutional Network
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.layer1 = GCNConv(in_channels, hidden_channels)
        self.layer2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = self.layer2(x, edge_index)
        return x
    
class ExtendedGCN(torch.nn.Module):
    def __init__(self, base_model, add_layer_dim, output_activation_function = torch.softmax):
        super(ExtendedGCN, self).__init__()
        self.base_model = base_model
        self.added_layer = GCNConv(base_model.out_channels, add_layer_dim)
        self.output_activation_function = output_activation_function

    def forward(self, x, edge_index):
        x = self.base_model(x, edge_index)
        x = self.output_activation_function(self.added_layer(x, edge_index), dim = 1)
        return x
    
class OCSVM:
    def __init__(self, data, kernel = 'rbf'):
        self.data = data.x.detach().numpy()
        self.P = data.P
        self.U = data.U
        self.kernel = kernel
    
    def train(self):
        self.model = OneClassSVM(gamma = 'auto', kernel = self.kernel)
        self.model.fit(self.data[self.P])

    def predict(self):
        return self.model.predict(self.data[self.U])