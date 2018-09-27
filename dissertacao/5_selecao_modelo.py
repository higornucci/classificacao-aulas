import warnings
import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8')
dados_completo.set_index('index', inplace=True)

conjunto_treinamento, conjunto_teste = train_test_split(dados_completo, test_size=0.2, random_state=42)
conjunto_treinamento = conjunto_treinamento[:48000]
conjunto_teste = conjunto_teste[-12000:]

X_treino, X_teste, Y_treino, Y_teste = conjunto_treinamento.drop('acabamento', axis=1), conjunto_teste.drop(
    'acabamento', axis=1), conjunto_treinamento['acabamento'], conjunto_teste['acabamento']
print('X Treino:', X_treino.head(10))
print('Y Treino:', Y_treino.head(10))
print('X Teste:', X_teste.head(10))
print('Y Teste:', Y_teste.head(10))
Y_teste.to_csv("y_teste.csv", encoding='utf-8', index=False)

seed = 7
num_folds = 5
processors = 3
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos_base = [('NB', MultinomialNB()),
                ('DTC', DecisionTreeClassifier()),
                ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())]


def rodar_algoritmos():
    global inicio, preds, final, grid_search
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    cv_resultados = cross_val_score(grid_search.best_estimator_, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros ' + nome + ' :', grid_search.best_estimator_)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    modelo.fit(X_treino, Y_treino)
    preds = modelo.predict(X_teste)
    final = time.time()


def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': [1, 5, 10, 20, 50],
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SVM':
        return [
            {'C': [.01, .1, 1, 10, 100],
             'gamma': [.0001, .001, .01, .1],
             'kernel': ['rbf', 'poly']}
        ]
    elif nome == 'DTC':
        return [
            {'max_features': [0, 1, 5, 10, 20, 31],
             'max_depth': [1, 5, 10, 15, 30],
             'class_weight': [None, 'balanced']
             }
        ]
    elif nome == 'NB':
        return [
            {'alpha': [0, .0001, .001, .01, .1, .5, 1, 2, 10, 20]}
        ]
    return None


def imprimir_resultados():
    resultado = pd.DataFrame()
    resultado["id"] = X_teste.index
    resultado["item.acabamento"] = preds
    resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))


# Validar cada um dos modelos
for nome, modelo in modelos_base:
    rodar_algoritmos()
    imprimir_resultados()
