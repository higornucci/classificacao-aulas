import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)
print(dados_completo.head())

random_state = 42
n_jobs = 3

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
    ('MNB', MultinomialNB(alpha=0.01)),
    ('RFC', RandomForestClassifier(random_state=random_state, class_weight='balanced', max_depth=50,
                                   max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=250,
                                   n_jobs=n_jobs)),
    ('K-NN', KNeighborsClassifier(n_neighbors=2, weights='distance')),
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
    pipeline = Pipeline([('bal', balanceador),
                         ('clf', modelo)])
    y_pred = cross_val_predict(pipeline, X_completo, Y_completo, cv=kfold, n_jobs=n_jobs)
    accuracy = accuracy_score(Y_completo, y_pred)
    print("Accuracy (train) for %s: %0.4f%% " % (nome, accuracy * 100))
    matriz_confusao = confusion_matrix(Y_completo, y_pred)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], False, title='Confusion matrix' + nome)
    plot_confusion_matrix(matriz_confusao, nome, [1, 2, 3, 4, 5], True,
                          title='Confusion matrix ' + nome + ', normalized')
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report(y_true=Y_completo, y_pred=y_pred, digits=4))
    print()


for nome, modelo in modelos_base:
    rodar_algoritmos()
