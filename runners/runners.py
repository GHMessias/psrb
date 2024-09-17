'''
This file contains all the functions that will be used to run the experiments
'''

from torch_geometric.data import Data
import torch
from utils.utils import *
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GAE, VGAE
from sklearn import svm
import pandas as pd
from networkx.algorithms import node_classification




def neg_inf(args, dataset_path, model_name, rates):

    '''
    Function to train the model to infer reliable negative examples
    '''
    for rate in rates:
        # organing the data in positive and unlabeled classes, setting the L graphs in case of rewiring

        # loading dataset
        dataset = torch.load(args.dataset_path, weights_only=False)
        dataset = Data(x = dataset[0]['x'], y = dataset[0]['y'], edge_index = dataset[0]['edge_index'])
        data = organize_data(data = dataset,
                                L = args.L,
                                rate = rate,
                                positive_class = args.positive_class,
                                name = args.dataset_name)
        
        
    return

def pu_classification(data, model):
    '''
    Função responsável por treinar os modelos a partir da segunda etapa, os dados negativos em data.N representam os elementos inferidos. 
    '''
    for element in data.N:
        data.infered_y[element] = 0

    weights_gcn = torch.tensor([ len(data.y[data.y == 1]) / len(data.y), len(data.y[data.y == 0])/ len(data.y)])
    weights_rgcn = torch.tensor([ len(data.y[data.y == 1]) / len(data.y),  len(data.y[data.y == 0])/ (len(data.y))])
    data.train_mask = torch.tensor([1 if data.infered_y[x] in [0,1] else 0 for x in range(data.num_nodes)], dtype = torch.bool)
    if isinstance(model, (CCRNE, MCLS, RCSVM)):
        clf = svm.SVC()
        try:
            clf.fit(data.x[data.train_mask].detach().numpy(), data.infered_y[data.train_mask].detach().numpy())
            y_pred = clf.predict(data.x[data.test_mask])
        except:
            return pd.DataFrame()
        return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1))
        # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)

    if isinstance(model, (VGAE, GAE)):
        if isinstance(model.encoder, RGCN):
            freeze_model_params(model.encoder)
            RGCN_classifier = ExtendedRGCN(model.encoder, 2, output_activation_function=torch.softmax)
            optimizer = torch.optim.Adam(RGCN_classifier.parameters(), lr = 0.01)
            
            criterion = torch.nn.CrossEntropyLoss(weight=weights_rgcn, reduction='mean')

            for epoch in range(200):
                optimizer.zero_grad()
                out = RGCN_classifier(data.x, data.graph_list)
                loss = criterion(out[data.train_mask], data.infered_y[data.train_mask])
                print(f'loss for pu classification: epoch {epoch} | loss {loss.item():.4f}', end = '\r')
                loss.backward()
                optimizer.step()

            RGCN_classifier.eval()
            y_pred = RGCN_classifier(data.x, data.graph_list)[data.test_mask]
            y_pred = torch.argmax(y_pred, dim = 1)
            return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred.detach().numpy(), pos_label = 1))
            # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)
        
        if isinstance(model.encoder, GCN):
            freeze_model_params(model.encoder)
            GCN_classifier = ExtendedGCN(model.encoder, 2).float()
            optimizer = torch.optim.Adam(GCN_classifier.parameters(), lr = 0.01)
            criterion = torch.nn.CrossEntropyLoss(weight = weights_gcn, reduction='mean')

            for epoch in range(200):
                optimizer.zero_grad()
                out = GCN_classifier(x = data.x, edge_index = data.edge_index)
                loss = criterion(out[data.train_mask], data.infered_y[data.train_mask])
                print(f'loss for pu classification: epoch {epoch} | loss {loss.item():.4f}', end = '\r')
                loss.backward()
                optimizer.step()

            GCN_classifier.eval()
            y_pred = GCN_classifier(data.x, data.edge_index)[data.test_mask]
            y_pred = torch.argmax(y_pred, dim = 1)
            return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred.detach().numpy(), pos_label = 1))
            # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)

    if isinstance(model, (LP_PUL, PU_LP)):
        G = to_networkx(data, to_undirected=True)
        labels = {node: label.item() for node, label in zip(G.nodes(), data.infered_y) if label in [0, 1]}
        nx.set_node_attributes(G, labels, 'label')
        y_pred = torch.tensor(node_classification.local_and_global_consistency(G))
        return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred[data.test_mask].detach().numpy()))
        # evaluate(data.y[data.test_mask].detach().numpy(), y_pred[data.test_mask].detach().numpy())

    if isinstance(model, OCSVM):
        return
    return