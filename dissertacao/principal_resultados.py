import sys
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo = dados_completo.sample(frac=1).reset_index(drop=True)
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)
dados_completo.drop(['other_incentives', 'total_area_confinement', 'area_20_erosion', 'quality_programs',
                     'lfi', 'fertigation', 'microrregiao#_BaixoPantanal'],
                    axis=1, inplace=True)
print(dados_completo.shape)
Y = dados_completo.pop('carcass_fatness_degree')
X = dados_completo

random_state = 42
n_jobs = 3

dados_completo_x, test_x, dados_completo_y, test_y = train_test_split(X, Y, test_size=0.2, stratify=Y,
                                                                      random_state=random_state)

dados_completo = dados_completo_x.join(dados_completo_y)
print(dados_completo.head())
print(dados_completo.shape)
dados_completo = []

enn = EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=5)
smote = SMOTE(n_jobs=n_jobs, random_state=random_state)
smoteenn = SMOTEENN(enn=EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=n_jobs), smote=SMOTE(n_jobs=n_jobs),
                    random_state=random_state)


class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


num_folds = 10
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)
rfc_coef = RandomForestClassifier(random_state=random_state, class_weight='balanced', max_depth=50,
                                  max_features='sqrt', min_samples_leaf=1, min_samples_split=6,
                                  n_estimators=250,
                                  n_jobs=n_jobs)


def fazer_selecao_features_rfe():
    pipeline = Mypipeline([('bal', enn),
                           ('clf', rfc_coef)])
    features = dados_completo_x.columns
    rfe = RFECV(estimator=pipeline, cv=kfold, scoring='f1_weighted', n_jobs=5)

    rfe.fit(dados_completo_x, dados_completo_y.values.ravel())
    print("Caraterísticas ordenadas pelo rank RFE:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))
    ranking = sorted(zip(rfe.support_, features))
    print("Características selecionadas", ranking)
    return rfe.transform(dados_completo_x)


# print(fazer_selecao_features_rfe())
# exit()

balanceadores = [
    ('ENN', enn),
    ('SMOTE', smote),
    ('SMOTEENN', smoteenn)
]

# preparando alguns modelos
modelos_base = [
    ('MNB', MultinomialNB(alpha=0.01)),
    ('RFC', RandomForestClassifier(random_state=random_state, class_weight='balanced', max_depth=50,
                                   max_features='sqrt', min_samples_leaf=1, min_samples_split=6, n_estimators=250,
                                   n_jobs=n_jobs)),

    ('ADA', AdaBoostClassifier(random_state=random_state, n_estimators=16)),
    ('MLP', MLPClassifier(random_state=random_state, activation='tanh', alpha=0.01, hidden_layer_sizes=14, solver='adam')),

    ('K-NN', KNeighborsClassifier(n_neighbors=2, weights='distance')),
    ('SVM', SVC(class_weight='balanced', C=128, gamma=8, kernel='rbf', random_state=random_state, probability=True))
]


def roc_auc_aux(y_test, y_pred_probas, nome, nome_balanceador):
    skplt.metrics.plot_roc(y_test, y_pred_probas)
    nome_arquivo = 'roc_auc_' + nome_balanceador + '_' + nome + '_best.png'
    plt.savefig('figuras/' + nome_arquivo)


def plot_confusion_matrix(cm, nome, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.gray):
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
    plt.tight_layout()
    plt.grid('off')
    plt.grid(None)
    plt.savefig('figuras/' + nome_arquivo)


def model_select():
    for nome_balanceador, balanceador in balanceadores:
        print(balanceador)
        pipeline = Pipeline([('bal', balanceador),
                             ('clf', modelo)])
        print("# Rodando o algoritmo %s" % nome)
        print()

        np.set_printoptions(precision=4)
        pipeline.fit(dados_completo_x, dados_completo_y)

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = pipeline.predict(test_x)
        matriz_confusao = confusion_matrix(test_y, y_pred)
        nome_arquivo = nome + '_' + nome_balanceador + '_best'
        plot_confusion_matrix(matriz_confusao, nome_arquivo, [1, 2, 3, 4, 5], False,
                              title='Confusion matrix' + nome + ' (best parameters)')
        plot_confusion_matrix(matriz_confusao, nome_arquivo, [1, 2, 3, 4, 5], True,
                              title='Confusion matrix ' + nome + ', normalized')
        print('Matriz de Confusão')
        print(matriz_confusao)
        print(classification_report(y_true=test_y, y_pred=y_pred, digits=4))
        y_pred = pipeline.predict_proba(test_x)
        roc_auc_aux(test_y, y_pred, nome, nome_balanceador)
        print()
        sys.stdout.flush()


for nome, modelo in modelos_base:
    print(modelo)
    model_select()
