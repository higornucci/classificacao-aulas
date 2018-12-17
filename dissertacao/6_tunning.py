import multiprocessing
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_predict, cross_val_score, \
    StratifiedKFold

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop(dados_completo.columns[0], axis=1, inplace=True)

random_state = 42
n_jobs = multiprocessing.cpu_count()  # - 1

classes_balancear = list([2, 3, 4])
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

# balanceador = ClusterCentroids(random_state=random_state)
# balanceador = RandomUnderSampler(random_state=random_state)
# balanceador = NearMiss(version=3)
# balanceador = AllKNN(allow_minority=True)
# balanceador = NeighbourhoodCleaningRule(n_jobs=n_jobs, sampling_strategy=list([2, 3, 4]))
balanceador = EditedNearestNeighbours(n_jobs=n_jobs, kind_sel='mode', sampling_strategy=classes_balancear)

# balanceador = SMOTE()
# balanceador = ADASYN()
# balanceador = RandomOverSampler()

# balanceador = SMOTEENN(random_state=random_state)
print(balanceador)
X_treino, Y_treino = balanceador.fit_resample(
    conjunto_treinamento.drop('acabamento', axis=1),
    conjunto_treinamento['acabamento'])
print(sorted(Counter(Y_treino).items()))
# X_treino, Y_treino = conjunto_treinamento.drop('acabamento', axis=1), conjunto_treinamento['acabamento']
X_treino = pd.DataFrame(data=X_treino, columns=X_completo.columns)

X_teste, Y_teste = conjunto_teste.drop('acabamento', axis=1), conjunto_teste['acabamento']
num_folds = 5
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)


# Tunando o melhor modelo
def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': range(12, 17, 1),
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [0.1, 1, 5],
             'C': [0.001, 100, 1000]
             #  },
             # {'kernel': ['sigmoid'],
             #  'gamma': [0.01, 0.1, 1, 5],
             #  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
             #  },
             # {'kernel': ['linear'],
             #  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
             }
        ]
    elif nome == 'DTC':
        return [
            {'max_features': range(20, 29, 1),
             'max_depth': [16, 17, 18, 19],
             'min_samples_split': range(5, 11, 1),
             'min_samples_leaf': range(1, 5, 1),
             # 'class_weight': [None, 'balanced']
             }
            # {'max_features': 20,
            #  'max_depth': 13,
            #  'min_samples_split': 7,
            #  'min_samples_leaf': 17,
            #  # 'class_weight': [None, 'balanced']
            #  }
        ]
    elif nome == 'MNB':
        return [
            {'alpha': [0.01, 1, 2, 3],
             'fit_prior': [True, False],
             'class_prior': [None, [1, 2, 3, 4, 5]]}
        ]
    elif nome == 'RF':
        return [
            {'n_estimators': range(10, 300, 50),
             'max_features': range(1, 27, 1),
             'max_depth': range(1, 10, 1),
             'min_samples_split': range(5, 10, 1),
             'min_samples_leaf': range(15, 20, 1)}
            # {'bootstrap': [False], 'n_estimators': [10, 50, 70], 'max_features': [10, 20, 27]}
        ]
    return None


def gerar_matriz_confusao(modelo, tipo, X_treino, Y_treino, X_teste, Y_teste):
    y_pred = cross_val_predict(modelo, X_treino, Y_treino, cv=3)
    matriz_confusao = confusion_matrix(Y_treino, y_pred)
    print('Matriz de Confusão ' + tipo)
    print(matriz_confusao)
    print(classification_report_imbalanced(Y_treino, y_pred))


def rodar_modelo(modelo, nome, tipo, X_treino, Y_treino, X_teste, Y_teste):
    cv_resultados = cross_val_score(modelo, X_treino, Y_treino, cv=kfold, scoring=scoring, n_jobs=n_jobs)
    print('Validação cruzada ' + nome + ' :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    gerar_matriz_confusao(modelo, tipo, X_treino, Y_treino, X_teste, Y_teste)
    return cv_resultados


def imprimir_acuracia(nome, df_results):
    plt.figure()
    sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
    sns.despine(top=True, right=True, left=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
    plt.xlabel('Acuracy')
    plt.ylabel('')
    plt.title('Difference in terms of acuracy with ' + nome)
    plt.savefig('acuracia' + nome + '.svg')


nome = 'MNB' #'DTC' 'K-NN' 'SVM'
modelo = None
grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
grid_search.fit(X_treino, Y_treino)
melhor_modelo = grid_search.best_estimator_
print('Melhores parametros ' + nome + ' :', melhor_modelo)
cv_results_balanced = rodar_modelo(melhor_modelo, nome, 'Balanceado', X_treino, Y_treino, X_teste, Y_teste)

df_results = (pd.DataFrame({'Balanced ': cv_results_balanced}).unstack().reset_index())
imprimir_acuracia(nome, df_results)
