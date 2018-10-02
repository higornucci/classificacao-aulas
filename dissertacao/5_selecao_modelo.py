import warnings
import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)


def mostrar_quantidade_por_classe(classe):
    print(dados_completo.loc[dados_completo['acabamento'] == classe].info())


# mostrar_quantidade_por_classe(1)
# mostrar_quantidade_por_classe(2)
# mostrar_quantidade_por_classe(3)
# mostrar_quantidade_por_classe(4)
# mostrar_quantidade_por_classe(5)


def buscar_quantidades_iguais(quantidade, classe):
    classe = dados_completo.loc[dados_completo['acabamento'] == classe]
    return classe.sample(quantidade, random_state=42)


def mostrar_correlacao(dados, classe):
    matriz_correlacao = dados.corr()
    print('Correlaçao com ' + classe + '\n', matriz_correlacao[classe].sort_values(ascending=False))


mostrar_correlacao(dados_completo, 'acabamento')

classe_1 = buscar_quantidades_iguais(2947, 1)
classe_2 = buscar_quantidades_iguais(5407, 2)
classe_3 = buscar_quantidades_iguais(5407, 3)
classe_4 = buscar_quantidades_iguais(5407, 4)
classe_5 = buscar_quantidades_iguais(198, 5)
dados_qtde_iguais = classe_2.append(classe_3).append(classe_4).append(classe_1).append(classe_5)

# conjunto_treinamento, conjunto_teste = train_test_split(dados_qtde_iguais, test_size=0.2, random_state=42)
# conjunt54hio_treinamento = conjunto_treinamento[:48000]
# conjunto_teste = conjunto_teste[-12000:]

X_treino, Y_treino = dados_qtde_iguais.drop('acabamento', axis=1), dados_qtde_iguais['acabamento']
# X_treino, X_teste, Y_treino, Y_teste = conjunto_treinamento.drop('acabamento', axis=1), conjunto_teste.drop(
#     'acabamento', axis=1), conjunto_treinamento['acabamento'], conjunto_teste['acabamento']
# print('X Treino:', X_treino.head(10))
# print('X Treino:', X_treino.info())
# print('Y Treino:', Y_treino.head(10))
# print('X Teste:', X_teste.head(10))
# print('Y Teste:', Y_teste.head(10))
# resultado = pd.DataFrame()
# resultado["id"] = Y_teste.index
# resultado["item.acabamento"] = Y_teste.values
# resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)

seed = 7
num_folds = 5
processors = 3
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos_base = [# ('NB', MultinomialNB()),
                # ('DTC', DecisionTreeClassifier()),
                # ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())]


def rodar_algoritmos():
    global inicio, final, grid_search  # , preds
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    cv_resultados = cross_val_score(grid_search.best_estimator_, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros ' + nome + ' :', grid_search.best_estimator_)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()


def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': [1, 4, 5, 10, 15],
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [0.1],
             'C': [1000, 2000]
             },
            # {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            # 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # },
            # {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # }
        ]
    elif nome == 'DTC':
        return [
            {'max_features': [1, 5, 8, 9, 10, 12, 20],
             'max_depth': [1, 5, 10, 11, 12, 13, 14],
             'class_weight': [None, 'balanced']
             }
        ]
    elif nome == 'NB':
        return [
            {'alpha': [0, .0001, .001, .01, .1, .5, 1, 5, 9, 10, 11, 15]}
        ]
    return None


def imprimir_resultados():
    # resultado = pd.DataFrame()
    # resultado["id"] = X_teste.index
    # resultado["item.acabamento"] = preds
    # resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))


# Validar cada um dos modelos
for nome, modelo in modelos_base:
    rodar_algoritmos()
    imprimir_resultados()
