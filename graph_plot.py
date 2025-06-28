import pandas as pd
import matplotlib.pyplot as plt

# Suponha que você já tenha o DataFrame `df` com as colunas: rate, model, F1

data_name = 'Cora'
df = pd.read_csv(f'results/neg_inf_results_{data_name}.csv')

# Agrupar por 'rate' e 'model', e calcular média e desvio padrão
summary = df.groupby(['rate', 'model'])['f1'].agg(['mean', 'std']).reset_index()

# Lista de modelos únicos
modelos = summary['model'].unique()

# Plot
plt.figure(figsize=(10, 6))

for model in modelos:
    dados_modelo = summary[summary['model'] == model]
    rates = dados_modelo['rate']
    medias = dados_modelo['mean']
    desvios = dados_modelo['std']
    
    # Linha principal (média do F1)
    plt.plot(rates, medias, label=model, marker = 'o')
    
    # Faixa do desvio padrão (transparente)
    plt.fill_between(rates, medias - desvios, medias + desvios, alpha=0.2)

plt.xlabel('Taxa de dados rotulados (rate)')
plt.ylabel('F1-score médio')
plt.title('Desempenho dos modelos com diferentes taxas de dados rotulados')
plt.legend(title='Modelo')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'plots/f1_score_neg_inf_{data_name}.png', dpi=300)
