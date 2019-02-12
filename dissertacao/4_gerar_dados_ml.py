import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

# Transformando tipos de dados de colunas numéricas
print(dados_completo.info())

dados_alvo = dados_completo['carcass_fatness_degree']
dados_alvo = pd.DataFrame(data=dados_alvo, columns=['carcass_fatness_degree'])
dados_numericos = dados_completo.drop('carcass_fatness_degree', axis=1)  # remover atributos não numéricos
dados_numericos_labels = dados_numericos.columns.values.tolist()
print('nomes colunas:', dados_numericos_labels)
dados_numericos[dados_numericos_labels] = MinMaxScaler().fit_transform(dados_numericos[dados_numericos_labels].values)
dados_numericos = pd.DataFrame(dados_numericos)
dados_numericos.columns = dados_numericos_labels

dados_completo = dados_numericos.join(dados_alvo)

print(dados_completo[dados_completo.isnull().any(axis=1)])
print(np.any(np.isnan(dados_completo)))
dados_completo.dropna(inplace=True)
dados_completo.to_csv('../input/DadosCompletoTransformadoML.csv', sep='\t')

print(dados_completo.info())
print(dados_completo.head())
