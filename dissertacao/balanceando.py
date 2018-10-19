import warnings
import numpy
import pandas as pd
from collections import Counter
from imblearn.under_sampling import ClusterCentroids
warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

cc = ClusterCentroids(random_state=42)
X_balanceado, Y_balanceado = cc.fit_resample(dados_completo.drop('acabamento', axis=1), dados_completo['acabamento'])
print(sorted(Counter(Y_balanceado).items()))
print(sorted(X_balanceado.shape()))
print(sorted(Y_balanceado.shape()))

dados_completo_balanceado = pd.DataFrame(numpy.concatenate(X_balanceado, Y_balanceado))

dados_completo_balanceado.to_csv('../input/DadosCompletoUnderBalanceadoML.csv', sep='\t')

print(dados_completo_balanceado.head())
print(dados_completo_balanceado.describe())
print(dados_completo_balanceado.info())
