import argparse
import torch
import random
from torch_geometric.utils import to_networkit
from rewiring.rewiring import rewiring
from models.models import *
from torch_geometric.nn import GAE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json


def parse_arguments():
    '''
    Function to collect the arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the JSON config file with parameters')
    parser.add_argument('--neg_inf', action='store_true', help='Train the model only to found reliable negatives')
    parser.add_argument('--model_names', type = str, nargs = '+', default = ['LP_PUL', 'PU_LP', 'MCLS', 'CCRNE', 'GCN', 'RGCN'], help = 'Models for experiment to run')
    parser.add_argument('--positive_class', type = int, default=1, help = 'Class that should be passed for positive examples. The rest of the classes will be considered as negative')
    parser.add_argument('--dataset_path', type = str, help = 'path that contais the data.pt file. The dataset need to follow the pytorch_goemetric.dataset structure.')
    parser.add_argument('--dataset_name', type = str, help = 'Name of the dataset. This will be used for name the experiments dataframe archive.')
    parser.add_argument('--sample', type = int, help = 'Number of samples to experiments to run', default = 1)
    parser.add_argument('--rates', nargs = '+', default = [0.01,0.5,0.1,0.2,0.25])
    parser.add_argument('--L', type = int, default = 2, help = 'Number of graphs to generate via rewiring')
    parser.add_argument('--neg_inf_only', type = bool, default = False, help = 'Parameter to determine if the model will only infer negative examples')
    parser.add_argument('--alpha', type = float, default = 0.2, help = 'Alpha parameter of the probability function')
    parser.add_argument('--beta', type = float, default = 0.5,help = 'Beta parameter of the probability function')
    parser.add_argument('--gamma', type = float, default = 2,help = 'Gamma parameter of the probability function')
    parser.add_argument('--hid_dim', type = int, default=64, help = 'Number of neurons in hidden dimension of neural net models')
    parser.add_argument('--out_dim', type = int, default=16, help = 'Number of neurons in output dimension of neural net models')

    return parser.parse_args()

def load_config_from_json(json_file):
    '''Function to load parameters from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def organize_data(data, L, rate, positive_class, alpha, beta, gamma, name):
    '''
    Function that organize the data object.

    Parameters:
    data (torch_geometric.data): torch_geometric.data object to organize
    L (int): parameter to apply the PSRB algorithm
    rate (float): rate of positive nodes to select at random
    positive_class (int): positive class to determine with object is positive. The remaining classes will be consedered as negatives
    name (str): name of the dataset
    alpha, beta, gamma (float): parameters of phi function
    seed_number (int): seed value, None by default

    Returns:
    Return the organized data object with the necessary positive and unlabeled elements.
    '''

    # setting data.y to be binary
    data.y = torch.tensor([1 if data.y[x] == positive_class else 0 for x in range(data.num_nodes)])

    # listing all the positive elements (this variable will not be used in the training phase of any algorithm)
    all_positives = [x for x in range(data.num_nodes) if data.y[x] == 1]

    # Positive and unlabeled elements
    data.P = torch.tensor(random.sample(all_positives , int(rate * len(all_positives))))
    data.U = torch.tensor([x for x in range(data.num_nodes) if x not in data.P])

    # List of graphs generated by the PSRB algorithm
    data.graph_list = rewiring(to_networkit(data.edge_index, directed=False), L, data.P, alpha=alpha, beta=beta, gamma=gamma)

    # Creating a tensor for reliable negatives and positive (hipothesis) elements
    data.infered_y = torch.tensor([-1] * data.num_nodes)
    for element in data.P:
        data.infered_y[element] = 1

    # Creating a test mask
    data.test_mask = torch.tensor([1 if x not in data.P else 0 for x in range(data.num_nodes)], dtype = torch.bool)

    # Naming the model
    data.name = name
    return data

def get_model(model_name, data, **kwargs):
    '''
    Function to return the model given the model_name. To add new models, please create a new class at models.models file.

    Parameter:
    model_name (str): String that represents the model
    data (pytorch.data.data): data object to create the model
    kwargs: L (int): number of iterations that the data contains for the rewiring method
            hid_dim (int): number of hidden neurons
            out_dim (int): number of output neurons
            activation_function (builtin_function_or_method): activation function (default torch.relu)

    Returns:
    model
    '''
    if model_name == 'CCRNE':
        return CCRNE(data)
    if model_name == 'LP_PUL':
        # return LP_PUL(data, graph = to_networkx(data, to_undirected = True))
        return LP_PUL(data)
    if model_name == 'MCLS':
        return MCLS(data)
    if model_name == 'OCSVM':
        return OCSVM(data)
    if model_name == 'PU_LP':
        return PU_LP(data)
    if model_name == 'RCSVM':
        return RCSVM(data)
    if model_name == 'RGCN':
        # return GAE(encoder = RGCN(data.x.shape[1], kwargs['hid_dim'], kwargs['out_dim'], kwargs['L'], kwargs['activation_function']))
        return GAE(encoder = RGCN(data.x.shape[1], kwargs['hid_dim'], kwargs['out_dim'], kwargs['L'], kwargs['activation_function']))
    if model_name == 'GCN':
        return GAE(encoder = GCN(data.x.shape[1], kwargs['hid_dim'], kwargs['out_dim']))

def gae_negative_inference(data, model, num_neg):
    inference_dict = dict()
    model.eval()
    if isinstance(model.encoder, RGCN):
        H_L = model.encode(data.x, data.graph_list)

    if isinstance(model.encoder, GCN):
        H_L = model.encode(data.x.float(), data.edge_index)

    for element in data.U:
        dist = torch.cdist(H_L[element].unsqueeze(0), H_L[data.P])
        value = dist.mean()
        inference_dict[element] = value
    dicionario_ordenado = dict(sorted(inference_dict.items(),reverse=True, key=lambda item: item[1]))
    return torch.stack(list(dicionario_ordenado.keys())[:num_neg])

def train_gae(data, gae_model, optimizer, epochs, verbose = False):
    if isinstance(gae_model.encoder, RGCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.graph_list)
            loss = gae_model.recon_loss(H_L, data.graph_list[-1].edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    if isinstance(gae_model.encoder, GCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.edge_index)
            loss = gae_model.recon_loss(H_L, data.edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    print('\n')
    return

def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

def evaluate(y_true, y_pred, pos_label = 1, verbose = True):
    acc =  round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred, pos_label = pos_label),4)
    recall = round(recall_score(y_true, y_pred, pos_label = pos_label),4)
    precision = round(precision_score(y_true, y_pred, pos_label=pos_label),4)
    try:
        _11 = confusion_matrix(y_true, y_pred)[1,1]
        _00 = confusion_matrix(y_true, y_pred)[0,0]
        _01 = confusion_matrix(y_true, y_pred)[0,1]
        _10 = confusion_matrix(y_true, y_pred)[1,0]
    except:
        _00, _11, _01, _10 = None, None, None, None

    if verbose:
        print(f'accuracy score: \n{acc}')
        print(f'f1 score: \n{f1}')
        print(f'recall score \n {recall}')
        print(f'precision score \n {precision}')
    return {'acc': [acc], 'f1': [f1], 'precision': [precision], 'recall': [recall], '1/1' : [_11], '0/0' : [_00], '0/1' : [_01], '1/0' : [_10]}