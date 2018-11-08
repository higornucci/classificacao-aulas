import warnings
import time
from collections import Counter
from functools import wraps

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold, \
    StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

random_state = 42

def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print('Elapsed computation time: {:.3f} secs'
              .format(elapsed_time))
        return elapsed_time, result

    return wrapper


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


def buscar_quantidades_iguais(quantidade, classe):
    classe = dados_completo.loc[dados_completo['acabamento'] == classe]
    return classe.sample(quantidade, random_state=7)


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

conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=random_state)
for trainamento_index, teste_index in split.split(dados_completo, dados_completo['acabamento']):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

# balanceador = ClusterCentroids(random_state=random_state)
# balanceador = RandomUnderSampler(random_state=random_state, replacement=True)
# balanceador = NearMiss(version=3)
balanceador = SMOTE()
# balanceador = ADASYN()
X_treino, Y_treino = balanceador.fit_resample(conjunto_treinamento.drop('acabamento', axis=1),
                                              conjunto_treinamento['acabamento'])
print(sorted(Counter(Y_treino).items()))

X_teste, Y_teste = conjunto_teste.drop('acabamento', axis=1), conjunto_teste['acabamento']

print('X Teste:', X_teste.info())
# mostrar_quantidade_por_classe(conjunto_treinamento, 1)
# mostrar_quantidade_por_classe(conjunto_treinamento, 2)
# mostrar_quantidade_por_classe(conjunto_treinamento, 3)
# mostrar_quantidade_por_classe(conjunto_treinamento, 4)
# mostrar_quantidade_por_classe(conjunto_treinamento, 5)
resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.acabamento"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)


@timeit
def fit_predict_imbalanced_model(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    n_correct = sum(y_pred == y_test)
    return n_correct / len(y_pred)


@timeit
def fit_predict_balanced_model(modelo, X_train, y_train, X_test, y_test):
    X_balanceado, y_balanceado = balanceador.fit_resample(X_train, y_train)
    modelo.fit(X_balanceado, y_balanceado)
    y_pred = modelo.predict(X_test)
    n_correct = sum(y_pred == y_test)
    return n_correct / len(y_pred)


def fazer_selecao_features():
    rfe = RFE(LogisticRegression(), 20)
    rfe = rfe.fit(X_treino, Y_treino)
    feature_rfe_scoring = pd.DataFrame({
        'feature': X_treino.columns,
        'score': rfe.ranking_
    })
    feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
    print('Features mais importantes: ', feat_rfe_20)


# fazer_selecao_features()

num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# preparando alguns modelos
modelos_base = [  # ('NB', MultinomialNB()),
    ('DTC', tree.DecisionTreeClassifier()),
    ('RF', RandomForestClassifier(random_state=random_state)),
    ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
    ('SVM', SVC())]


def gerar_matriz_confusao(modelo):
    average = 'weighted'
    y_train_pred = cross_val_predict(modelo, X_treino, Y_treino, cv=kfold)
    matriz_confusao = confusion_matrix(Y_treino, y_train_pred)
    print('Matriz de Confusão')
    print(matriz_confusao)


def rodar_algoritmos():
    global preds
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    print('Melhores parametros ' + nome + ' :', melhor_modelo)

    skf = StratifiedKFold(n_splits=num_folds, random_state=random_state)
    X = conjunto_treinamento.drop('acabamento', axis=1)
    Y = conjunto_treinamento['acabamento']

    cv_results_imbalanced = []
    cv_time_imbalanced = []
    cv_results_balanced = []
    cv_time_balanced = []
    for train_idx, valid_idx in skf.split(X, Y):
        # X_local_train = preprocessor.fit_transform(X_train.iloc[train_idx])
        X_local_train = X.iloc[train_idx]
        y_local_train = Y.iloc[train_idx].values.ravel()
        # X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
        X_local_test = X.iloc[valid_idx]
        y_local_test = Y.iloc[valid_idx].values.ravel()

        elapsed_time, score = fit_predict_imbalanced_model(melhor_modelo, X_local_train, y_local_train, X_local_test,
                                                             y_local_test)
        cv_time_imbalanced.append(elapsed_time)
        cv_results_imbalanced.append(score)

        elapsed_time, score = fit_predict_balanced_model(melhor_modelo, X_local_train, y_local_train, X_local_test,
                                                           y_local_test)
        cv_time_balanced.append(elapsed_time)
        cv_results_balanced.append(score)
        print('Resultados não balanceado ', cv_results_imbalanced)
        print('Resultados balanceado ', cv_results_balanced)

    # mostrar_features_mais_importantes(melhor_modelo)
    gerar_matriz_confusao(melhor_modelo)

    df_results = (pd.DataFrame({'Balanced model': cv_results_balanced,
                                'Imbalanced model': cv_results_imbalanced})
                  .unstack().reset_index())
    df_time = (pd.DataFrame({'Balanced model': cv_time_balanced,
                             'Imbalanced model': cv_time_imbalanced})
               .unstack().reset_index())

    melhor_modelo.fit(X_treino, Y_treino)
    preds = melhor_modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))

    plt.figure()
    sns.boxplot(y='level_0', x=0, data=df_time)
    sns.despine(top=True, right=True, left=True)
    plt.xlabel('time [s]')
    plt.ylabel('')
    plt.title('Computation time difference using a random under-sampling')

    plt.figure()
    sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
    sns.despine(top=True, right=True, left=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
    plt.xlabel('ROC-AUC')
    plt.ylabel('')
    plt.title('Difference in terms of ROC-AUC using a random under-sampling')


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
            {'n_neighbors': [15],
             'weights': ['uniform']}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [5],
             'C': [1],
             'class_weight': ['balanced']
             }
            # {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            # 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # },
            # {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # }
        ]
    elif nome == 'DTC':
        return [
            # {'max_features': [1, 10, 13, 20, 27],
            # 'max_depth': [1, 10, 15, 16, 17],
            # 'min_samples_split': range(10, 100, 5),
            # 'min_samples_leaf': range(1, 30, 2),
            # 'class_weight': [None, 'balanced']
            # }
            {'max_features': [20],
             'max_depth': [13],
             'min_samples_split': [7],
             'min_samples_leaf': [17],
             # 'class_weight': [None, 'balanced']
             }
        ]
    elif nome == 'NB':
        return [
            {'alpha': range(5, 10, 1),
             'fit_prior': [True, False],
             'class_prior': [None, [1, 2, 3, 4, 5]]}
        ]
    elif nome == 'RF':
        return [
            {'n_estimators': range(10, 300, 50),
             'max_features': [10, 20, 27],
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
    rodar_algoritmos()
    imprimir_resultados()
