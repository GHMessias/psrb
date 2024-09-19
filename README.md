# Positive Sequential Rewiring via Breadth First Search (PSRB)

Esse repositório é a implementação do modelo PSRB juntamente com os experimentos propostos no artigo.

⚠️⚠️⚠️ REPOSITÓRIO EM CONSTRUÇÃO ⚠️⚠️⚠️

## Sobre

O modelo PSRB consiste na aplicação de uma função de probabilidade geometrica para criação de arestas entre vértices positivos em um problema de Positive and Unlabeled Learning (PUL).

## Utilização

O PSRB opera em duas formas distintas: (1) Utilização de comand lines ou (2) usando um json de configuração (recomenda-se a utilização de um json dada a facilidade de editar os parâmetros)

```
python3 main.py [options]
```

|Option|Domain|Default|Description|
|------|------|-------|-----------|
|--config|str|.|Path to json file with configs|
|--neg_inf||||
|--model_names|list(str)|['LP_PUL', 'PU_LP', 'MCLS', 'CCRNE', 'GCN', 'RGCN']|List with names of the models|
|--positive_class|int|1|Positive class to consider in PUL approach|
|--dataset_path|str|.|Path of data.pt file of dataset|
|--dataset_name|str|.|Name of the dataset to generate output file|
|--sample|int|10|number of samples to run|
|--rates|list|[0.01,0.5,0.1,0.2,0.25]|rates of positive labels to test the model|
|--L|int|2|number of iterations (Graphs) generated by the probability density function|
|--alpha|float|0.2|Value alpha of probability density function|
|--beta|float|0.5|Value beta of probability density function|
|--gamma|float|2|Value gamma of probability density function|
|--hid_dim|int|64|value of first layer of GAE|
|--out_dim|int|16|value of last layer of GAE|
