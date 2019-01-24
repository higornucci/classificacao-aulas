import itertools
import multiprocessing
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)

random_state = 42
n_jobs = multiprocessing.cpu_count() - 2


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
    plt.savefig(nome_arquivo)


classes_balancear = list([2, 3])
print('Classes para balancear', classes_balancear)
test_size = 0.2
train_size = 0.8
print(((train_size * 100), '/', test_size * 100))
X_completo = dados_completo.drop(['acabamento'], axis=1)
Y_completo = dados_completo['acabamento']
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
for trainamento_index, teste_index in split.split(X_completo, Y_completo):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='mode',
                                      sampling_strategy=classes_balancear, n_neighbors=4)
print(balanceador)
X_treino = pd.read_csv('../input/DadosCompletoTransformadoMLBalanceadoX.csv', encoding='utf-8', delimiter='\t')
X_treino.drop(X_treino.columns[0], axis=1, inplace=True)
Y_treino = pd.read_csv('../input/DadosCompletoTransformadoMLBalanceadoY.csv', encoding='utf-8', delimiter='\t')
Y_treino.drop(Y_treino.columns[0], axis=1, inplace=True)
print(sorted(Counter(Y_treino).items()))

X_teste, Y_teste = conjunto_teste.drop('acabamento', axis=1), conjunto_teste['acabamento']
num_folds = 4
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

param_grid = {"n_estimators": [250],
              "min_samples_leaf": [1, 3],
              'max_features': ['sqrt', 9, 10, 11],
              'max_depth': [50, 75]}

scores = ['accuracy', 'f1_macro']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    np.set_printoptions(precision=4)
    clf = GridSearchCV(RandomForestClassifier(random_state=random_state, oob_score=True),
                       param_grid, cv=kfold, n_jobs=n_jobs, scoring=score, verbose=2)
    clf.fit(X_treino, Y_treino.values.ravel())

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")


    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_teste, clf.predict(X_teste)
    matriz_confusao = confusion_matrix(Y_teste, y_pred)
    plot_confusion_matrix(matriz_confusao, 'RFC_' + score, [1, 2, 3, 4, 5], False,
                          title='Confusion matrix RFC (best parameters)')
    plot_confusion_matrix(matriz_confusao, 'RFC_' + score, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + 'RFC' + ', normalized')
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report_imbalanced(y_true, y_pred))
    print()