import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompleto.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

# Substituindo os valores do acabamento
dados_completo['acabamento'].replace(['Magra - Gordura ausente'], 1, inplace=True)
dados_completo['acabamento'].replace(['Gordura Escassa - 1 a 3 mm de espessura'], 2, inplace=True)
dados_completo['acabamento'].replace(['Gordura Mediana - acima de 3 a até 6 mm de espessura'], 3, inplace=True)
dados_completo['acabamento'].replace(['Gordura Uniforme - acima de 6 e até 10 mm de espessura'], 4, inplace=True)
dados_completo['acabamento'].replace(['Gordura Excessiva - acima de 10 mm de espessura'], 5, inplace=True)

# Substituindo os valores da tipificação
dados_completo['tipificacao'].replace(['Macho INTEIRO'], '0', inplace=True)  # M
dados_completo['tipificacao'].replace(['Macho CASTRADO'], '1', inplace=True)  # C
dados_completo['tipificacao'].replace(['Fêmea'], '2', inplace=True)  # F

# Substituindo os valores da maturidade
dados_completo['maturidade'].replace(['Dente de leite'], 0, inplace=True)
dados_completo['maturidade'].replace(['Dois dentes'], 2, inplace=True)
dados_completo['maturidade'].replace(['Quatro dentes'], 4, inplace=True)
dados_completo['maturidade'].replace(['Seis dentes'], 6, inplace=True)
dados_completo['maturidade'].replace(['Oito dentes'], 8, inplace=True)


def eh_precoce(linha):
    if linha['maturidade'] == 0:
        return 1
    if linha['maturidade'] == 2:
        return 1
    if linha['maturidade'] == 4 and linha['tipificacao'] != 0:
        return 1
    return 0


# Substituindo os valores da rispoa
# rispoa_label_encoder = LabelEncoder()
# rispoa_labels = rispoa_label_encoder.fit_transform(dados_completo['rispoa'])
# dados_completo['rispoa'] = rispoa_labels
# rispoa_mapeamento = {index: label for index, label in enumerate(rispoa_label_encoder.classes_)}
# print(rispoa_mapeamento)

# Substituindo os valores da tipificacao
# tipificacao_label_encoder = LabelEncoder()
# tipificacao_labels = tipificacao_label_encoder.fit_transform(dados_completo['tipificacao'])
# dados_completo['tipificacao'] = tipificacao_labels
# tipificacao_mapeamento = {index: label for index, label in enumerate(tipificacao_label_encoder.classes_)}
# print(tipificacao_mapeamento)

# Substituindo os valores 'Não' por 0
dados_completo = dados_completo.applymap(lambda x: 0 if "Não" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 0 if "NÃO" in str(x) else x)

# Substituindo os valores 'Sim' por 0
dados_completo = dados_completo.applymap(lambda x: 1 if "Sim" in str(x) else x)
dados_completo = dados_completo.applymap(lambda x: 1 if "SIM" in str(x) else x)

dados_completo.to_csv('../input/DadosCompletoTransformado.csv', sep='\t')
