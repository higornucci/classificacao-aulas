import os
import itertools
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo = dados_completo.sample(frac=1).reset_index(drop=True)
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)
print(dados_completo.head())

random_state = 42
n_jobs = 3

classes_balancear = list([2, 3])
print('Classes para balancear', classes_balancear)
balanceador = EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=5)
# balanceador = SMOTEENN()
# balanceador = SMOTE(n_jobs=n_jobs)
print(balanceador)
X_completo, Y_completo = dados_completo.drop('carcass_fatness_degree', axis=1), \
                         dados_completo['carcass_fatness_degree']

num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

# preparando alguns modelos
modelos_base = [
    # ('MNB', MultinomialNB(alpha=0.01)),
    ('RFC', RandomForestClassifier(random_state=random_state, class_weight='balanced', max_depth=50,
                                   max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=250,
                                   n_jobs=n_jobs)),
    # ('K-NN', KNeighborsClassifier(n_neighbors=2, weights='distance')),
    ('SVM', SVC(class_weight='balanced', C=50, gamma=2, kernel='rbf'))
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


def rodar_algoritmos():
    i = 1
    y_pred_all = np.ndarray(Y_completo.shape, dtype=int)
    for train_index, test_index in kfold.split(X_completo, Y_completo):
        print('Treinando ' + nome)
        print("iteration", i, ":")
        x_local_train, x_local_test = X_completo.iloc[train_index], X_completo.iloc[test_index]
        y_local_train, y_local_test = Y_completo.iloc[train_index], Y_completo.iloc[test_index]

        if os.path.isfile('../input/DadosCompletoTransformadoMLBalanceadoX2Treino' + str(i) + '.csv'):
            x_local_test, x_local_train, y_local_test, y_local_train = abrir_conjuntos(i)
        else:
            x_local_test, x_local_train, y_local_test, y_local_train = salvar_conjuntos(i, x_local_test, x_local_train,
                                                                                        y_local_test, y_local_train)

        modelo.fit(x_local_train, y_local_train)
        y_pred = modelo.predict(x_local_test)
        accuracy = accuracy_score(y_local_test, y_pred)
        print("Accuracy (train) for %d: %0.4f%% " % (i, accuracy * 100))
        i = i + 1
        y_pred_all[test_index] = y_pred
        matriz_confusao = confusion_matrix(y_local_test, y_pred)
        print('Matriz de Confusão')
        print(matriz_confusao)
        print(classification_report(y_true=y_local_test, y_pred=y_pred, digits=4))
        print()
        print("=====================================")
        sys.stdout.flush()
    return y_pred_all


def abrir_conjuntos(i):
    x_treino_nome = '../input/DadosCompletoTransformadoMLBalanceadoX2Treino' + str(i) + '.csv'
    y_treino_nome = '../input/DadosCompletoTransformadoMLBalanceadoY2Treino' + str(i) + '.csv'
    x_teste_nome = '../input/DadosCompletoTransformadoMLX2Teste' + str(i) + '.csv'
    y_teste_nome = '../input/DadosCompletoTransformadoMLY2Teste' + str(i) + '.csv'
    x_local_train = pd.read_csv(x_treino_nome, encoding='utf-8', delimiter='\t')
    x_local_train.drop(x_local_train.columns[0], axis=1, inplace=True)
    y_local_train = pd.read_csv(y_treino_nome, encoding='utf-8', delimiter='\t')
    y_local_train.drop(y_local_train.columns[0], axis=1, inplace=True)
    x_local_test = pd.read_csv(x_teste_nome, encoding='utf-8', delimiter='\t')
    x_local_test = x_local_test.set_index(list(x_local_test.columns[[0]]))
    y_local_test = pd.read_csv(y_teste_nome, encoding='utf-8', delimiter='\t')
    y_local_test = y_local_test.set_index(list(y_local_test.columns[[0]]))
    return x_local_test, x_local_train, y_local_test, y_local_train


def salvar_conjuntos(i, x_local_test, x_local_train, y_local_test, y_local_train):
    x_treino_nome = '../input/DadosCompletoTransformadoMLBalanceadoX2Treino' + str(i) + '.csv'
    y_treino_nome = '../input/DadosCompletoTransformadoMLBalanceadoY2Treino' + str(i) + '.csv'
    x_teste_nome = '../input/DadosCompletoTransformadoMLX2Teste' + str(i) + '.csv'
    y_teste_nome = '../input/DadosCompletoTransformadoMLY2Teste' + str(i) + '.csv'

    x_local_train = pd.DataFrame(data=x_local_train, columns=X_completo.columns)
    y_local_train = pd.DataFrame(data=y_local_train, columns=['carcass_fatness_degree'])
    x_local_test = pd.DataFrame(data=x_local_test, columns=X_completo.columns)
    y_local_test = pd.DataFrame(data=y_local_test, columns=['carcass_fatness_degree'])

    x_local_train, y_local_train = balanceador.fit_resample(x_local_train, y_local_train)

    x_local_train = pd.DataFrame(data=x_local_train, columns=X_completo.columns)
    y_local_train = pd.DataFrame(data=y_local_train, columns=['carcass_fatness_degree'])

    x_local_train.to_csv(x_treino_nome, encoding='utf-8', sep='\t')
    y_local_train.to_csv(y_treino_nome, encoding='utf-8', sep='\t')
    x_local_test.to_csv(x_teste_nome, encoding='utf-8', sep='\t')
    y_local_test.to_csv(y_teste_nome, encoding='utf-8', sep='\t')
    return x_local_test, x_local_train, y_local_test, y_local_train


for nome, modelo in modelos_base:
    y_predicao = rodar_algoritmos()
    # y_predicao = pd.Series(y_predicao)
    matriz_confusao = confusion_matrix(Y_completo, y_predicao)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], False,
                          title='Confusion matrix' + nome)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + nome + ', normalized')
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report(y_true=Y_completo, y_pred=y_predicao, digits=4))
    print()
    sys.stdout.flush()
