import warnings
import pandas as pd

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('estabelecimento_identificador', inplace=True)

colunas_categoricas = [
    'identificador_lote_situacao_lote', 'tipificacao', 'estabelecimento_municipio', 'estabelecimento_uf',
    'pratica_recuperacao_pastagem_outra_pratica', 'organizacao_estabelecimento_pertence']

for cc in colunas_categoricas:
    dummies = pd.get_dummies(dados_completo[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    dados_completo.drop(cc, axis=1, inplace=True)
    dados_completo = dados_completo.join(dummies)

dados_completo.to_csv('../input/DadosCompletoTransformadoDummies.csv', sep='\t')
