import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)


def plot_resampling(X, y, ax):
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y)


def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=7)
for trainamento_index, teste_index in split.split(dados_completo, dados_completo['acabamento']):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

# balanceador = ClusterCentroids(random_state=0)
balanceador = SMOTE()
# balanceador = ADASYN()
X_balanceado, Y_balanceado = balanceador.fit_resample(conjunto_treinamento.drop('acabamento', axis=1),
                                                      conjunto_treinamento['acabamento'])
print(sorted(Counter(Y_balanceado).items()))
print(X_balanceado)
print(Y_balanceado)

X_teste, Y_teste = conjunto_teste.drop(
    'acabamento', axis=1), conjunto_teste['acabamento']

resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.acabamento"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)

kfold = StratifiedKFold(n_splits=5, random_state=7)
average = 'weighted'


def gerar_matriz_confusao_treino(modelo):
    y_train_pred = cross_val_predict(modelo, X_balanceado, Y_balanceado, cv=kfold)
    matriz_confusao = confusion_matrix(Y_balanceado, y_train_pred)
    print('Matriz de Confusão Treino')
    print(matriz_confusao)
    precision = precision_score(Y_balanceado, y_train_pred, average=average)
    print('Precision: ', precision)
    recall = recall_score(Y_balanceado, y_train_pred, average=average)
    print('Recall: ', recall)


# grid_search = GridSearchCV(SVC(), [{'kernel': ['rbf'],
#                                     'gamma': [0.01, 0.1, 1, 5],
#                                     'C': [1, 100, 1000],
#                                     'class_weight': ['balanced', None]
#                                     },
#                                    {'kernel': ['sigmoid'],
#                                     'gamma': [0.01, 0.1, 1, 5],
#                                     'C': [1, 100, 1000],
#                                     },
#                                    {'kernel': ['linear'],
#                                     'C': [1, 100, 1000],
#                                     }],
#                            cv=kfold,
#                            n_jobs=-1)
# grid_search.fit(X_balanceado, Y_balanceado)
# svc_clf = grid_search.best_estimator_
svc_clf = SVC(C=1000, class_weight='balanced', gamma=0.01, kernel='rbf')
print('Melhores parametros :', svc_clf)
clf = svc_clf
clf.fit(X_balanceado, Y_balanceado)
gerar_matriz_confusao_treino(clf)

# plot_decision_function(X_balanceado, Y_balanceado, clf, ax1)
# ax1.set_title('Decision function for {}'.format(balanceador.__class__.__name__))
# plot_resampling(X_balanceado, Y_balanceado, balanceador, ax2)
# ax2.set_title('Resampling using {}'.format(balanceador.__class__.__name__))
# fig.tight_layout()

y_pred = svc_clf.predict(X_teste)
matriz_confusao = confusion_matrix(Y_teste, y_pred)
print('Matriz de Confusão Teste')
print(matriz_confusao)
precision = precision_score(Y_teste, y_pred, average=average)
print('Precision: ', precision)
recall = recall_score(Y_teste, y_pred, average=average)
print('Recall: ', recall)

resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.acabamento"] = y_pred
resultado.to_csv("y_pred.csv", encoding='utf-8', index=False)
