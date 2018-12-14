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
n_jobs = multiprocessing.cpu_count() #- 1


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


def buscar_quantidades_iguais(quantidade, classe):
    classe = dados_completo.loc[dados_completo['acabamento'] == classe]
    return classe.sample(quantidade, random_state=random_state)


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
classes_balancear = list([3])
print('Classes para balancear', classes_balancear)
test_size = 0.3
train_size = 0.7
print(((train_size * 100), '/', test_size * 100))
X_completo = dados_completo.drop(['acabamento'], axis=1)
Y_completo = dados_completo['acabamento']
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
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
# X_treino, Y_treino = conjunto_treinamento.drop('acabamento', axis=1), conjunto_treinamento['acabamento']
X_treino = pd.DataFrame(data=X_treino, columns=X_completo.columns)

X_teste, Y_teste = conjunto_teste.drop('acabamento', axis=1), conjunto_teste['acabamento']

print('X Treino:', X_treino.head(50))
print('Y Treino:', Y_treino)
# mostrar_quantidade_por_classe(conjunto_treinamento, 1)
# mostrar_quantidade_por_classe(conjunto_treinamento, 2)
# mostrar_quantidade_por_classe(conjunto_treinamento, 3)
# mostrar_quantidade_por_classe(conjunto_treinamento, 4)
# mostrar_quantidade_por_classe(conjunto_treinamento, 5)
resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.classe"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)


def fazer_selecao_features_vt():
    features = X_treino.columns
    vt = VarianceThreshold(threshold=(.8 * (1 - .8)))
    vt.fit(X_treino)
    print("Caraterísticas ordenadas pelo rank VarianceThreshold:")
    print(X_treino[features[vt.get_support(indices=True)]])
    print(sorted(zip(map(lambda x: round(x, 4), vt.variances_), features), reverse=True))


def fazer_selecao_features_rfe():
    features = X_treino.columns
    rfe = RFE(tree.DecisionTreeClassifier(), 22)
    rfe.fit(X_treino, Y_treino)
    print("Caraterísticas ordenadas pelo rank RFE:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))
    print("Características selecionadas", sorted(zip(rfe.support_, features)))


def pretty_print_linear(coefs, names=None, sort=False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                      for coef, name in lst)


def fazer_selecao_features_lasso():
    features = X_treino.columns
    lasso = Lasso(alpha=.3)
    lasso.fit(X_treino, Y_treino)
    print("Lasso model: ", pretty_print_linear(lasso.coef_, features, sort=True))


fazer_selecao_features_vt()
fazer_selecao_features_rfe()
fazer_selecao_features_lasso()

num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# preparando alguns modelos
modelos_base = [('MNB', MultinomialNB()),
                ('DTC', tree.DecisionTreeClassifier()),
                # ('RF', RandomForestClassifier(random_state=random_state)),
                ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())
                ]


def gerar_matriz_confusao(modelo, tipo, X_treino, Y_treino, X_teste, Y_teste):
    average = None
    modelo.fit(X_treino, Y_treino)
    y_pred = modelo.predict(X_teste)
    matriz_confusao = confusion_matrix(Y_teste, y_pred)
    print('Matriz de Confusão ' + tipo)
    print(matriz_confusao)
    precision = precision_score(Y_teste, y_pred, average=average)
    print('Precision: ', precision)
    recall = recall_score(Y_teste, y_pred, average=average)
    print('Recall: ', recall)
    f1 = f1_score(Y_teste, y_pred, average=average)
    print('F1 score: ', f1)
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
    global preds
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    print('Melhores parametros ' + nome + ' :', melhor_modelo)
    cv_results_balanced = rodar_modelo(melhor_modelo, nome, 'Balanceado', X_treino, Y_treino, X_teste, Y_teste)
    cv_results_imbalanced = rodar_modelo(melhor_modelo, nome, 'Não Balanceado',
                                         conjunto_treinamento.drop('acabamento', axis=1),
                                         conjunto_treinamento['acabamento'], X_teste, Y_teste)

    melhor_modelo.fit(X_treino, Y_treino)
    preds = melhor_modelo.predict(X_teste)
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


def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': range(12, 17, 1),
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [0.1, 1, 5],
             'C': [0.001, 100, 1000]
             #  },
             # {'kernel': ['sigmoid'],
             #  'gamma': [0.01, 0.1, 1, 5],
             #  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
             #  },
             # {'kernel': ['linear'],
             #  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
             }
        ]
    elif nome == 'DTC':
        return [
            {'max_features': range(20, 29, 1),
             'max_depth': [16, 17, 18, 19],
             'min_samples_split': range(5, 11, 1),
             'min_samples_leaf': range(1, 5, 1),
             # 'class_weight': [None, 'balanced']
             }
            # {'max_features': 20,
            #  'max_depth': 13,
            #  'min_samples_split': 7,
            #  'min_samples_leaf': 17,
            #  # 'class_weight': [None, 'balanced']
            #  }
        ]
    elif nome == 'MNB':
        return [
            {'alpha': [0.01, 1, 2, 3],
             'fit_prior': [True, False],
             'class_prior': [None, [1, 2, 3, 4, 5]]}
        ]
    elif nome == 'RF':
        return [
            {'n_estimators': range(10, 300, 50),
             'max_features': range(1, 27, 1),
             'max_depth': range(1, 10, 1),
             'min_samples_split': range(5, 10, 1),
             'min_samples_leaf': range(15, 20, 1)}
            # {'bootstrap': [False], 'n_estimators': [10, 50, 70], 'max_features': [10, 20, 27]}
        ]
    return None


def imprimir_resultados():
    resultado = pd.DataFrame()
    resultado["id"] = X_teste.index
    resultado["item.acabamento"] = preds
    resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)


for nome, modelo in modelos_base:
    cv_results_balanced, cv_results_imbalanced = rodar_algoritmos()
    df_results = (pd.DataFrame({'Balanced ': cv_results_balanced,
                                'Imbalanced ': cv_results_imbalanced})
                  .unstack().reset_index())
    imprimir_acuracia(nome, df_results)
    # imprimir_resultados()
