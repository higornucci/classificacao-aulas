import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompleto.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

# Substituindo os valores do acabamento
dados_completo['carcass_fatness_degree'].replace(['Magra - Gordura ausente'], 1, inplace=True)
dados_completo['carcass_fatness_degree'].replace(['Gordura Escassa - 1 a 3 mm de espessura'], 2, inplace=True)
dados_completo['carcass_fatness_degree'].replace(['Gordura Mediana - acima de 3 a até 6 mm de espessura'], 3, inplace=True)
dados_completo['carcass_fatness_degree'].replace(['Gordura Uniforme - acima de 6 e até 10 mm de espessura'], 4, inplace=True)
dados_completo['carcass_fatness_degree'].replace(['Gordura Excessiva - acima de 10 mm de espessura'], 5, inplace=True)

colunas_categoricas = ['typification', 'maturity', 'microrregiao', 'mesoregiao', 'mes_abate', 'estacao_abate']
dados_categoricos = dados_completo[colunas_categoricas]
dados_alvo = dados_completo['carcass_fatness_degree']
dados_numericos = dados_completo.drop(colunas_categoricas, axis=1).drop('carcass_fatness_degree', axis=1)  # remover atributos não numéricos

for cc in colunas_categoricas:
    prefix = '{}#'.format(cc)
    dummies = pd.get_dummies(dados_categoricos[cc], prefix=prefix).astype(np.int8)
    dados_categoricos.drop(cc, axis=1, inplace=True)
    dados_categoricos = dados_categoricos.join(dummies)

dados_completo = dados_categoricos.join(dados_numericos).join(dados_alvo)

# Substituindo os valores 'Não' por 0
dados_completo = dados_completo.applymap(lambda x: 0 if "Não" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 0 if "NÃO" in str(x) else x)

# Substituindo os valores 'Sim' por 0
dados_completo = dados_completo.applymap(lambda x: 1 if "Sim" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 1 if "SIM" in str(x) else x)
print(np.any(np.isnan(dados_completo)))

dados_completo.to_csv('../input/DadosCompletoTransformado.csv', sep='\t')
