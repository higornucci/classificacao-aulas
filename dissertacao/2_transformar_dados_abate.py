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

# Substituindo os valores da tipificação
dados_completo['typification'].replace(['Macho INTEIRO'], 0, inplace=True)  # M
dados_completo['typification'].replace(['Macho CASTRADO'], 1, inplace=True)  # C
dados_completo['typification'].replace(['Fêmea'], 2, inplace=True)  # F

# Substituindo os valores da maturidade
dados_completo['maturity'].replace(['Dente de leite'], 0, inplace=True)
dados_completo['maturity'].replace(['Dois dentes'], 2, inplace=True)
dados_completo['maturity'].replace(['Quatro dentes'], 4, inplace=True)
dados_completo['maturity'].replace(['Seis dentes'], 6, inplace=True)
dados_completo['maturity'].replace(['Oito dentes'], 8, inplace=True)


def eh_precoce(linha):
    if linha['maturity'] == 0:
        return 1
    if linha['maturity'] == 2:
        return 1
    if linha['maturity'] == 4 and linha['typification'] != 0:
        return 1
    return 0


colunas_categoricas = ['microrregiao', 'mesoregiao']
dados_categoricos = dados_completo[colunas_categoricas]
for cc in colunas_categoricas:
    labelEncoder = LabelEncoder()
    dados_categoricos[cc] = labelEncoder.fit_transform(dados_categoricos[cc])
    print(cc, labelEncoder.classes_)

dados_alvo = dados_completo['carcass_fatness_degree']
dados_completo = dados_completo.drop(colunas_categoricas, axis=1).drop('carcass_fatness_degree', axis=1)
dados_completo = dados_completo.join(dados_categoricos).join(dados_alvo)

# Substituindo os valores 'Não' por 0
dados_completo = dados_completo.applymap(lambda x: 0 if "Não" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 0 if "NÃO" in str(x) else x)

# Substituindo os valores 'Sim' por 0
dados_completo = dados_completo.applymap(lambda x: 1 if "Sim" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 1 if "SIM" in str(x) else x)
print(np.any(np.isnan(dados_completo)))

dados_completo.to_csv('../input/DadosCompletoTransformado.csv', sep='\t')
