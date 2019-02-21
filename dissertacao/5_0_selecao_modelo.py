import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
print(dados_completo.head())

random_state = 42
n_jobs = 2

# classes_balancear = list([2, 3])
# print('Classes para balancear', classes_balancear)
# balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='all',
#                                       sampling_strategy=classes_balancear, n_neighbors=3)
balanceador = SMOTEENN()
# balanceador = SMOTE(n_jobs=n_jobs)

X_completo, Y_completo = dados_completo.drop('carcass_fatness_degree', axis=1), \
                     dados_completo['carcass_fatness_degree']

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


def gerar_matriz_confusao(Y_completo, y_pred):
    matriz_confusao = confusion_matrix(Y_completo, y_pred)
    print('Matriz de Confusão')
    print(matriz_confusao)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + nome + ', normalized')
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], False, title='Confusion matrix ' + nome)
    print(classification_report(y_true=Y_completo, y_pred=y_pred, digits=4))


def rodar_algoritmos():
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', modelo)])
    grid_search = GridSearchCV(pipeline, escolher_parametros(), cv=kfold, n_jobs=n_jobs)
    grid_search.fit(X_completo, Y_completo)
    melhor_modelo = grid_search.best_estimator_
    y_pred = cross_val_predict(melhor_modelo, X_completo, Y_completo, cv=kfold, n_jobs=n_jobs)

    print('Melhores parametros ' + nome + ' :', melhor_modelo)
    gerar_matriz_confusao(Y_completo, y_pred)


def escolher_parametros():
    if nome == 'K-NN':
        return [{'clf__weights': ['uniform', 'distance'],
                 'clf__n_neighbors': [1, 2, 3, 4, 5, 10, 15, 20]}]
    elif nome == 'SVM':
        return [{'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
                 'clf__gamma': [0.001, 0.01, 0.1, 1, 10],
                 'clf__kernel': ['rbf']}]
    elif nome == 'MNB':
        return [{'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)}]
    elif nome == 'RFC':
        return [{'clf__n_estimators': [100, 250],
                 'clf__min_samples_leaf': [1, 5, 10],
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__max_features': ['sqrt', 'log2', None],
                 'clf__criterion': ['gini', 'entropy'],
                 'clf__class_weight': ['balanced', None],
                 'clf__max_depth': [50, 75]}
                ]
    return None


for nome, modelo in modelos_base:
    rodar_algoritmos()
