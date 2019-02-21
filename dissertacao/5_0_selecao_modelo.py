import itertools
import warnings
import time
import multiprocessing
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
print(dados_completo.head())

random_state = 42
n_jobs = 2

conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()

# classes_balancear = list([2, 3])
# print('Classes para balancear', classes_balancear)
# balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='all',
#                                       sampling_strategy=classes_balancear, n_neighbors=3)
balanceador = SMOTEENN()
# balanceador = SMOTE(n_jobs=n_jobs)

X_treino, Y_treino = conjunto_treinamento.drop('carcass_fatness_degree', axis=1), \
                     conjunto_treinamento['carcass_fatness_degree']

X_teste, Y_teste = conjunto_teste.drop('carcass_fatness_degree', axis=1), \
                   conjunto_teste['carcass_fatness_degree']

resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.classe"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)


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
modelos_base = [
    ('MNB', MultinomialNB()),
    ('RFC', RandomForestClassifier(random_state=random_state, oob_score=True)),
    ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
    ('SVM', SVC(gamma='scale'))
]


def plot_confusion_matrix(cm, nome, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.gray):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        nome_arquivo = 'matriz_confusao_normalizada_' + nome + '.png'
    else:
        nome_arquivo = 'matriz_confusao_' + nome + '.png'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.grid('off')
    plt.tight_layout()
    plt.savefig('figuras/' + nome_arquivo)


def gerar_matriz_confusao(modelo, nome, tipo, X_treino, Y_treino):
    modelo.fit(X_treino, Y_treino.values.ravel())
    y_pred = modelo.predict(X_teste)
    matriz_confusao = confusion_matrix(Y_teste, y_pred)
    print('Matriz de Confusão ' + tipo)
    print(matriz_confusao)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + nome + ', normalized')
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], False, title='Confusion matrix ' + nome)
    print(classification_report(y_true=Y_teste, y_pred=y_pred, digits=4))


def rodar_algoritmos():
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', modelo)])
    grid_search = GridSearchCV(pipeline, escolher_parametros(), cv=kfold, n_jobs=n_jobs)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring, n_jobs=n_jobs)

    print('Melhores parametros ' + nome + ' :', melhor_modelo)
    print('Validação cruzada ' + nome + ' :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    gerar_matriz_confusao(melhor_modelo)


def escolher_parametros():
    if nome == 'K-NN':
        return [{'clf__n_neighbors': range(13, 17, 2),
                 'clf__weights': ['uniform', 'distance']}]
    elif nome == 'SVM':
        return [{'clf__kernel': ['rbf'],
                 'clf__gamma': [5],  # 0.01, 0.1, 1, 5],
                 'clf__C': [1000]  # 0.001, 0.10, 0.1, 10, 25, 50, 100,
                 #  },
                 # {'kernel': ['sigmoid'],
                 # 'gamma': [0.01, 0.1, 1, 5],
                 # 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
                 # },
                 # {'kernel': ['linear'],
                 #              'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
                 }]
    elif nome == 'MNB':
        return [{'clf__alpha': range(1, 10, 1),
                 'clf__fit_prior': [True, False],
                 'clf__class_prior': [None, [1, 2, 3, 4, 5]]}]
    elif nome == 'RFC':
        return [{'clf__n_estimators': range(10, 300, 50),
                 'clf__max_features': range(1, 27, 2),
                 'clf__max_depth': range(1, 10, 1),
                 'clf__min_samples_split': range(5, 10, 1),
                 'clf__min_samples_leaf': range(15, 20, 1)}
                # {'bootstrap': [False], 'n_estimators': [10, 50, 70], 'max_features': [10, 20, 27]}
                ]
    return None


for nome, modelo in modelos_base:
    rodar_algoritmos()
