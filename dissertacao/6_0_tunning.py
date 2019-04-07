import itertools
import multiprocessing
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo = dados_completo.sample(frac=1).reset_index(drop=True)
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)

random_state = 42
n_jobs = 2


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


# classes_balancear = list([2, 3])
# print('Classes para balancear', classes_balancear)
test_size = 0.2
train_size = 0.8
print(((train_size * 100), '/', test_size * 100))
X_completo = dados_completo.drop(['carcass_fatness_degree'], axis=1)
Y_completo = dados_completo['carcass_fatness_degree']
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
for trainamento_index, teste_index in split.split(X_completo, Y_completo):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

balanceador = EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=5)
# balanceador = SMOTEENN()
print(balanceador)

X_treino, Y_treino = conjunto_treinamento.drop('carcass_fatness_degree', axis=1), \
                     conjunto_treinamento['carcass_fatness_degree']
X_teste, Y_teste = conjunto_teste.drop('carcass_fatness_degree', axis=1), \
                   conjunto_teste['carcass_fatness_degree']
num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# param_grid = {'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)}
# modelo = MultinomialNB()

# param_grid = {'clf__n_estimators': [100, 250],
#               'clf__min_samples_leaf': [1, 5, 10],
#               'clf__min_samples_split': [2, 5, 10],
#               'clf__max_features': ['sqrt', 'log2', None],
#               'clf__class_weight': ['balanced'],
#               'clf__max_depth': [50, 75]}
# modelo = RandomForestClassifier(oob_score=True)

param_grid = {'clf__C': [0.03125, 0.125, 0.5, 2, 8, 16],
              'clf__gamma': [2**-5, 2**-3, 2**-1, 2**1, 2**3],
              'clf__kernel': ['rbf']}
modelo = SVC()

# param_grid = {'clf__weights': ['uniform', 'distance'],
#               'clf__n_neighbors': [1, 2, 3, 4, 5, 10, 15, 20]}
# modelo = KNeighborsClassifier()

pipeline = Pipeline([('bal', balanceador),
                     ('clf', modelo)])

scores = ['f1_weighted', 'precision_weighted', 'recall_weighted']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    np.set_printoptions(precision=4)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kfold,
                               n_jobs=n_jobs, scoring=score, verbose=2)
    grid_search.fit(X_treino, Y_treino.values.ravel())

    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.4f (+/-%0.04f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_teste, grid_search.predict(X_teste)
    np.set_printoptions(precision=4)
    matriz_confusao = confusion_matrix(Y_teste, y_pred)
    plot_confusion_matrix(matriz_confusao, 'SVC_' + score, [1, 2, 3, 4, 5], False,
                          title='Confusion matrix SVC (best parameters)')
    plot_confusion_matrix(matriz_confusao, 'SVC_' + score, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + 'SVC' + ', normalized')
    print('Matriz de Confusao')
    np.set_printoptions(precision=4)
    print(matriz_confusao)
    print(classification_report(y_true, y_pred))
    print()
    sys.stdout.flush()
