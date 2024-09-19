# Positive Sequential Rewiring via Breadth First Search (PSRB)

Esse repositório é a implementação do modelo PSRB juntamente com os experimentos propostos no artigo.

⚠️⚠️⚠️ REPOSITÓRIO EM CONSTRUÇÃO ⚠️⚠️⚠️

## Sobre

O modelo PSRB consiste na aplicação de uma função de probabilidade geometrica para criação de arestas entre vértices positivos em um problema de Positive and Unlabeled Learning (PUL).

## Utilização

O PSRB opera em duas formas distintas: (1) Utilização de comand lines ou (2) usando um json de configuração (recomenda-se a utilização de um json dada a facilidade de editar os parâmetros)

´´´ python3 main.py [options] ´´´

|Option|Domain|Default|Description|
|------|------|-------|-----------|
|--config|str||Path to json file with configs|