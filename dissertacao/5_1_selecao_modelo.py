import warnings
import time
from collections import Counter
from functools import wraps

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import BinaryEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN, NeighbourhoodCleaningRule, \
    EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold, \
    StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Imputer, MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, SGDClassifier

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformado2.csv', encoding='utf-8', delimiter='\t')
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


def transformar_dados_colunas(X_train):
    num_pipeline = Pipeline([
        ('std_scaler', MinMaxScaler()),
    ])
    full_pipeline = ColumnTransformer(transformers=[
        ("num", num_pipeline, [19, 20]),
        ("cat", OneHotEncoder(), [0, 1, 18])],
        remainder='passthrough')
    X_train = full_pipeline.fit_transform(X_train)
    return pd.DataFrame(X_train)


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


def transformar_dados_colunas(X_train):
    num_pipeline = Pipeline([
        ('std_scaler', MinMaxScaler()),
    ])
    full_pipeline = ColumnTransformer(transformers=[
        ("num", num_pipeline, [19, 20]),
        ("cat", BinaryEncoder(), [0, 1, 18])],
        remainder='passthrough')
    X_train = full_pipeline.fit_transform(X_train)
    return pd.DataFrame(X_train)


# mostrar_correlacao(dados_completo, 'acabamento')

conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=random_state)
for trainamento_index, teste_index in split.split(dados_completo, dados_completo['acabamento']):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

# balanceador = ClusterCentroids(random_state=random_state)
# balanceador = RandomUnderSampler(random_state=random_state)
# balanceador = NearMiss(version=3)
# balanceador = AllKNN(allow_minority=True)
balanceador = EditedNearestNeighbours(sampling_strategy='auto', n_jobs=-1)
# balanceador = NeighbourhoodCleaningRule(sampling_strategy='auto', n_jobs=-1)

# balanceador = SMOTE()
# balanceador = ADASYN()

# balanceador = SMOTEENN(random_state=random_state)
X_treino, Y_treino = balanceador.fit_resample(
    transformar_dados_colunas(conjunto_treinamento.drop('acabamento', axis=1)),
    conjunto_treinamento['acabamento'])
print(sorted(Counter(Y_treino).items()))

X_teste, Y_teste = transformar_dados_colunas(conjunto_teste.drop('acabamento', axis=1)), conjunto_teste['acabamento']

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
    X_train, y_train = EditedNearestNeighbours(sampling_strategy='auto', n_jobs=-1).fit_resample(X_train, y_train)
    # bbc = BalancedBaggingClassifier(base_estimator=modelo,
    #                                 sampling_strategy='all',
    #                                 replacement=False,
    #                                 random_state=random_state)
    # bbc.fit(X_train, y_train)
    # y_pred = bbc.predict(X_test)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    n_correct = sum(y_pred == y_test)
    return n_correct / len(y_pred)
    # return balanced_accuracy_score(y_test, y_pred)


def fazer_selecao_features():
    rfe = RFE(LogisticRegression(), 20)
    rfe = rfe.fit(X_treino, Y_treino)
    feature_rfe_scoring = pd.DataFrame({
        'feature': X_treino.columns,
        'score': rfe.ranking_
    })
    feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
    print('Features mais importantes: ', feat_rfe_20)


num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# preparando alguns modelos
modelos_base = [
    # ('GNB', GaussianNB()),
    ('MNB', MultinomialNB()),
    # ('DTC', tree.DecisionTreeClassifier()),
    # ('RF', RandomForestClassifier(random_state=random_state)),
    # ('SGD', SGDClassifier(random_state=random_state, class_weight='balanced')),
    # ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
    # ('SVM', SVC())
]


def gerar_matriz_confusao(modelo):
    y_train_pred = cross_val_predict(modelo, X_treino, Y_treino, cv=kfold)
    matriz_confusao = confusion_matrix(Y_treino, y_train_pred)
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report_imbalanced(Y_treino, y_train_pred))


def rodar_algoritmos():
    global preds
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    print('Melhores parametros ' + nome + ' :', melhor_modelo)

    skf = StratifiedKFold(n_splits=num_folds, random_state=random_state)
    X = transformar_dados_colunas(conjunto_treinamento.drop('acabamento', axis=1))
    Y = conjunto_treinamento['acabamento']
    # X = X_treino
    # Y = Y_treino

    cv_results_imbalanced = []
    cv_time_imbalanced = []
    cv_results_balanced = []
    cv_time_balanced = []
    for train_idx, valid_idx in skf.split(X, Y):
        X_local_train = X.iloc[train_idx]
        y_local_train = Y.iloc[train_idx]
        # X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
        X_local_test = X.iloc[valid_idx]
        y_local_test = Y.iloc[valid_idx]

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

    df_results = (pd.DataFrame({'Balanced ': cv_results_balanced,
                                'Imbalanced ': cv_results_imbalanced})
                  .unstack().reset_index())
    df_time = (pd.DataFrame({'Balanced ': cv_time_balanced,
                             'Imbalanced ': cv_time_imbalanced})
               .unstack().reset_index())

    final = time.time()
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))

    plt.figure()
    sns.boxplot(y='level_0', x=0, data=df_time)
    sns.despine(top=True, right=True, left=True)
    plt.xlabel('time [s]')
    plt.ylabel('')
    plt.title('Computation time difference with ' + nome)
    plt.savefig('tempo ' + nome + '.svg')

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


def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': range(10, 17, 1),
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SGD':
        return [
            {'alpha': [10 ** x for x in range(-6, 1)],  # learning rate
             'n_iter': [1000],  # number of epochs
             'loss': ['log'],  # logistic regression,
             'penalty': ['l2', 'elasticnet'],
             'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
             'n_jobs': [-1]}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [1, 5, 25],
             'C': [250, 1000, 2000],
             'class_weight': ['balanced', '']
             }
        ]
    elif nome == 'DTC':
        return [
            # {'max_features': [1, 10, 13, 20, 27],
            # 'max_depth': [1, 10, 15, 16, 17],
            # 'min_samples_split': range(10, 100, 5),
            # 'min_samples_leaf': range(1, 30, 2),
            # 'class_weight': [None, 'balanced']
            # }
            {'max_features': range(15, 29, 1),
             'max_depth': range(5, 15, 1),
             'min_samples_split': range(5, 10, 1),
             'min_samples_leaf': range(10, 20, 1),
             # 'class_weight': [None, 'balanced']
             }
        ]
    elif nome == 'MNB':
        return [
            {'alpha': [0.001, 0.01, 0.1, 1, 4, 5, 6, 7],
             'fit_prior': [True, False],
             'class_prior': [None, [1, 2, 3, 4, 5]]}
        ]
    elif nome == 'GNB':
        return [
            {'var_smoothing': [1e-09, 1e-05, 1]}
        ]
    elif nome == 'RF':
        return [
            {'n_estimators': range(10, 300, 50),
             'max_features': [10, 20, 25, 29],
             'max_depth': range(1, 10, 1),
             'min_samples_split': range(5, 10, 1),
             'min_samples_leaf': range(15, 20, 1)}
        ]
    return None


for nome, modelo in modelos_base:
    rodar_algoritmos()
    # imprimir_resultados()
