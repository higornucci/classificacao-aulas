import pandas as pd

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('estabelecimento_identificador', inplace=True)
