# coding=utf-8
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)


def detect_outliers(series, whis=1.5):
    q75, q25 = np.percentile(series, [75, 25])
    iqr = q75 - q25
    return ~((series - series.median()).abs() <= (whis * iqr))


dados = pd.read_csv('../input/campos_precoce_dados_teste.csv')

# categorizando as colunas
dados.set_index('id', inplace=True)
dados["estabelecimento.cidade"] = dados["estabelecimento.cidade"].astype('object')
dados["item.sexo"] = dados["item.sexo"].astype('category')

resultado_desconhecido = dados['item.acabamento'].isnull()
dados_copia = dados.copy()
Y_copia = dados_copia['item.acabamento'].copy()

# removendo colunas irrelevantes
dados_copia.drop('estabelecimento.uf', axis=1, inplace=True)  # sempre igual
dados_copia.drop('item.peso', axis=1, inplace=True)  # o peso depois do abate não é relevante
dados_copia.drop('item.acabamento', axis=1, inplace=True)  # coluna Y
dados_copia.drop('item.data_abate', axis=1, inplace=True)  # data do abate é a mesma
dados_copia.drop('item.tipo_produto', axis=1, inplace=True)  # mesmo que item.acabamento
dados_copia.drop('item.sigla_doenca', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.confinamento_respostas', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.fabricao_racao', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.pratica_recuperacao_pastagem.descricao_outra_pratica', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.pratica_recuperacao_pastagem.descricao', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.incentivo.situação', axis=1, inplace=True)  # todos os valores nulos
dados_copia.drop('questionario.incentivo.descricao', axis=1, inplace=True)  # todos os valores nulos

colunas_categoricas = ['estabelecimento.cidade', 'item.sexo']

for cc in colunas_categoricas:
    dummies = pd.get_dummies(dados_copia[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    dados_copia.drop(cc, axis=1, inplace=True)
    dados_copia = dados_copia.join(dummies)

dados_envio = dados_copia[resultado_desconhecido]

X = dados_copia[~resultado_desconhecido]
Y = Y_copia[~resultado_desconhecido]

# mostrando o tamanho dos conjuntos
print('Dataset limpo: {}'.format(dados_copia.shape))
print('Dataset de treino: {}'.format(X.shape))
print('Dataset das classes: {}'.format(Y.shape))

seed = 7
num_folds = 3
processors = 1
scoring = 'roc_auc'
kfold = KFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos = [('SVM', LinearSVC(C=1.0)),
           ('K-NN', KNeighborsClassifier(n_neighbors=5)),
           ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)),
           ('NB', MultinomialNB())]

# Validar cada um dos modelos
for nome, modelo in modelos:
    cv_resultados = cross_val_score(modelo, X, Y, cv=kfold, n_jobs=1)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    modelo.fit(X, Y)
    preds = modelo.predict(dados_envio)
    resultado = pd.DataFrame()
    resultado["id"] = dados_envio.index
    resultado["item.acabamento"] = preds
    resultado.to_csv("resultado_" + nome + ".csv", index=False)
