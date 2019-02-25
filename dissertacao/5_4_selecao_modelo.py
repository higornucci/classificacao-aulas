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
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from yellowbrick.features import RFECV

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)
print(dados_completo.head())

random_state = 42
n_jobs = 1

classes_balancear = list([2, 3])
print('Classes para balancear', classes_balancear)
# balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='all',
#                                       sampling_strategy=classes_balancear, n_neighbors=3)
# balanceador = SMOTEENN()
balanceador = SMOTE(n_jobs=n_jobs)
print(balanceador)
X_completo, Y_completo = dados_completo.drop('carcass_fatness_degree', axis=1), \
                     dados_completo['carcass_fatness_degree']

num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)


def fazer_selecao_features_rfe():
    features = X_completo.columns
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', MultinomialNB())])
    rfe = RFECV(pipeline, cv=kfold, scoring='recall_weighted')

    rfe.fit(X_completo, Y_completo.values.ravel())
    print(rfe.poof())
    print("Caraterísticas ordenadas pelo rank RFE:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))
    ranking = sorted(zip(rfe.support_, features))
    print("Características selecionadas", ranking)
    return rfe.transform(X_completo)


# print(fazer_selecao_features_rfe())
# exit()
# preparando alguns modelos
modelos_base = [
    ('MNB', MultinomialNB()),
    ('RFC', RandomForestClassifier(random_state=random_state, oob_score=True,
                            n_estimators=100, class_weight='balanced')),
    ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
    ('SVM', SVC(gamma='scale', class_weight='balanced'))
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


def rodar_algoritmos_predict():
    print('Treinando ' + nome)
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', modelo)])
    y_pred = cross_val_predict(pipeline, X_completo, Y_completo, cv=kfold, n_jobs=n_jobs)
    matriz_confusao = confusion_matrix(Y_completo, y_pred)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], False,
                          title='Confusion matrix' + nome)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + nome + ', normalized')
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report(y_true=Y_completo, y_pred=y_pred, digits=4))
    print()


def rodar_algoritmos_score():
    print('Treinando ' + nome)
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', modelo)])
    scores = cross_val_score(pipeline, X_completo, Y_completo, cv=kfold, n_jobs=n_jobs)
    print(scores)
    print('Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))


for nome, modelo in modelos_base:
    rodar_algoritmos_score()
