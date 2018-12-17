import warnings
import time
import multiprocessing
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold, \
    StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import RFE, VarianceThreshold
from yellowbrick.features import RFECV
from sklearn.linear_model import LogisticRegression, RandomizedLasso, Lasso, Ridge

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

# 1 Iris Setosa, 2 Iris Versicolour, 3 Iris Virginica
# dados_completo = pd.read_csv('../input/iris.csv', encoding='utf-8', delimiter=',')
dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)
# dados_completo = pd.read_csv('../input/dados.csv', encoding='utf-8', delimiter='\t')
print(dados_completo.head())

random_state = 42
n_jobs = multiprocessing.cpu_count()  # - 1


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


def mostrar_correlacao(dados, classe):
    matriz_correlacao = dados.corr()
    print('Correlaçao com ' + classe + '\n', matriz_correlacao[classe].sort_values(ascending=False))

    colunas = ['C', 'F', 'M', 'mat', 'peso', '% class', 'out_inc', 'fab_rac', 'area_conf', 'area_man_80_cob',
               'area_man_20_er', 'id_ind', 'sisbov', 'cont_past', 'lita_trace', 'atest_prog_quali', 'envolvido_org',
               'confi', 'semi_confi', 'suple', 'ferti', 'ifp', 'ilp', 'ilpf', 'lat', 'lon', 'prec', 'acab']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matriz_correlacao, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 28, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(colunas)
    ax.set_yticklabels(colunas)
    plt.xticks(rotation=90)
    plt.savefig('corr.svg')
    plt.show()


# mostrar_correlacao(dados_completo, 'acabamento')
classes_balancear = list([2, 3])
print('Classes para balancear', classes_balancear)
test_size = 0.2
train_size = 0.8
print(((train_size * 100), '/', test_size * 100))
X_completo = dados_completo.drop(['acabamento'], axis=1)
Y_completo = dados_completo['acabamento']
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=random_state)
for trainamento_index, teste_index in split.split(X_completo, Y_completo):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

# balanceador = ClusterCentroids(random_state=random_state)
# balanceador = RandomUnderSampler(random_state=random_state)
# balanceador = NearMiss(version=3)
# balanceador = AllKNN(allow_minority=True)
# balanceador = NeighbourhoodCleaningRule(n_jobs=n_jobs, sampling_strategy=list([2, 3, 4]))
balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='mode', sampling_strategy=classes_balancear)

# balanceador = SMOTE()
# balanceador = ADASYN()
# balanceador = RandomOverSampler()

# balanceador = SMOTEENN(random_state=random_state)
print(balanceador)
X_treino, Y_treino = balanceador.fit_resample(
    conjunto_treinamento.drop('acabamento', axis=1),
    conjunto_treinamento['acabamento'])
print(sorted(Counter(Y_treino).items()))
X_treino = pd.DataFrame(data=X_treino, columns=X_completo.columns)

X_teste, Y_teste = conjunto_teste.drop('acabamento', axis=1), conjunto_teste['acabamento']

print('X Treino:', X_treino.head(50))
print('X Treino', X_treino.shape)
print('Y Treino:', Y_treino)
resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.classe"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)


def fazer_selecao_features_rfe():
    features = X_treino.columns
    rfe = RFECV(RandomForestClassifier(), cv=kfold, scoring='f1_weighted')

    rfe.fit(X_treino, Y_treino)
    print(rfe.poof())
    print("Caraterísticas ordenadas pelo rank RFE:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))
    ranking = sorted(zip(rfe.support_, features))
    print("Características selecionadas", ranking)
    return rfe.transform(X_treino)


# print(fazer_selecao_features_rfe())
# exit()
num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# preparando alguns modelos
modelos_base = [('MNB', MultinomialNB()),
                ('DTC', tree.DecisionTreeClassifier()),
                ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())
                ]


def gerar_matriz_confusao(modelo, tipo, X_treino, Y_treino, X_teste, Y_teste):
    modelo.fit(X_treino, Y_treino)
    y_pred = modelo.predict(X_teste)
    matriz_confusao = confusion_matrix(Y_teste, y_pred)
    print('Matriz de Confusão ' + tipo)
    print(matriz_confusao)
    print(classification_report_imbalanced(Y_teste, y_pred))


def rodar_modelo(modelo, nome, tipo, X_treino, Y_treino, X_teste, Y_teste):
    cv_resultados = cross_val_score(modelo, X_treino, Y_treino, cv=kfold, scoring=scoring, n_jobs=n_jobs)
    print('Validação cruzada ' + nome + ' :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    gerar_matriz_confusao(modelo, tipo, X_treino, Y_treino, X_teste, Y_teste)
    return cv_resultados


def imprimir_acuracia(nome, df_results):
    plt.figure()
    sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
    sns.despine(top=True, right=True, left=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
    plt.xlabel('Acuracy')
    plt.ylabel('')
    plt.title('Difference in terms of acuracy with ' + nome)
    plt.savefig('acuracia' + nome + '.svg')


def rodar_algoritmos():
    inicio = time.time()
    cv_results_balanced = rodar_modelo(modelo, nome, 'Balanceado', X_treino, Y_treino, X_teste, Y_teste)
    cv_results_imbalanced = rodar_modelo(modelo, nome, 'Não Balanceado',
                                         conjunto_treinamento.drop('acabamento', axis=1),
                                         conjunto_treinamento['acabamento'], X_teste, Y_teste)

    final = time.time()
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))
    return cv_results_balanced, cv_results_imbalanced


def mostrar_features_mais_importantes(melhor_modelo):
    if nome == 'RF':
        print('Características mais importantes RF :')
        feature_importances = pd.DataFrame(melhor_modelo.feature_importances_,
                                           index=X_treino.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)


for nome, modelo in modelos_base:
    cv_results_balanced, cv_results_imbalanced = rodar_algoritmos()
    df_results = (pd.DataFrame({'Balanced ': cv_results_balanced,
                                'Imbalanced ': cv_results_imbalanced})
                  .unstack().reset_index())
    imprimir_acuracia(nome, df_results)
