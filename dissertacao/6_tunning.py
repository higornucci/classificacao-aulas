import multiprocessing
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)

random_state = 42
n_jobs = multiprocessing.cpu_count()  # - 1

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
num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.001, 0.0001],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=kfold,
                       scoring='%s_macro' % score, refit=True, verbose=2)
    clf.fit(X_treino, Y_treino)

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
    print('Matriz de Confusão')
    print(matriz_confusao)
    print(classification_report_imbalanced(y_true, y_pred))
    print()
