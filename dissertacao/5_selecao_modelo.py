import warnings
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


def buscar_quantidades_iguais(quantidade, classe):
    classe = dados_completo.loc[dados_completo['acabamento'] == classe]
    return classe.sample(quantidade, random_state=7)


def mostrar_correlacao(dados, classe):
    matriz_correlacao = dados.corr()
    print('Correlaçao com ' + classe + '\n', matriz_correlacao[classe].sort_values(ascending=False))

    colunas = ['C', 'F', 'M', 'mat', 'peso', '% class', 'out_inc', 'fab_rac', 'area_conf', 'area_man_80_cob',
               'area_man_20_er', 'id_ind', 'sisbov', 'cont_past', 'lita_trace', 'atest_prog_quali', 'envolvido_org',
               'confi', 'semi_confi', 'suple', 'ferti', 'ifp', 'ilp', 'ilpf', 'lat', 'lon', 'prec', 'acab']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matriz_correlacao, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, 28, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(colunas)
    ax.set_yticklabels(colunas)
    plt.xticks(rotation=90)
    plt.savefig('corr.svg')
    plt.show()


mostrar_correlacao(dados_completo, 'acabamento')


conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=7)
for trainamento_index, teste_index in split.split(dados_completo, dados_completo['acabamento']):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

X_treino, X_teste, Y_treino, Y_teste = conjunto_treinamento.drop('acabamento', axis=1), conjunto_teste.drop(
    'acabamento', axis=1), conjunto_treinamento['acabamento'], conjunto_teste['acabamento']
print('X Treino:', X_treino.info())
print('X Teste:', X_teste.info())
mostrar_quantidade_por_classe(conjunto_treinamento, 1)
mostrar_quantidade_por_classe(conjunto_treinamento, 2)
mostrar_quantidade_por_classe(conjunto_treinamento, 3)
mostrar_quantidade_por_classe(conjunto_treinamento, 4)
mostrar_quantidade_por_classe(conjunto_treinamento, 5)
resultado = pd.DataFrame()
resultado["id"] = Y_teste.index
resultado["item.acabamento"] = Y_teste.values
resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)


def fazer_selecao_features():
    rfe = RFE(LogisticRegression(), 20)
    rfe = rfe.fit(X_treino, Y_treino)
    feature_rfe_scoring = pd.DataFrame({
        'feature': X_treino.columns,
        'score': rfe.ranking_
    })
    feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
    print('Features mais importantes: ', feat_rfe_20)


fazer_selecao_features()

seed = 7
num_folds = 10
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos_base = [('NB', MultinomialNB()),
                ('DTC', tree.DecisionTreeClassifier()),
                ('RF', RandomForestClassifier(random_state=seed)),
                ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())]


def gerar_matriz_confusao(modelo):
    average = 'weighted'
    y_train_pred = cross_val_predict(modelo, X_treino, Y_treino, cv=kfold)
    matriz_confusao = confusion_matrix(Y_treino, y_train_pred)
    print('Matriz de Confusão')
    print(matriz_confusao)
    precision = precision_score(Y_treino, y_train_pred, average=average)
    print('Precision: ', precision)
    recall = recall_score(Y_treino, y_train_pred, average=average)
    print('Recall: ', recall)


def rodar_algoritmos():
    global preds
    inicio = time.time()
    grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    cv_resultados = cross_val_score(BaggingClassifier(melhor_modelo), X_treino, Y_treino, cv=kfold, scoring=scoring)

    mostrar_features_mais_importantes(melhor_modelo)
    gerar_matriz_confusao(melhor_modelo)

    print('Melhores parametros ' + nome + ' :', melhor_modelo)
    print('Validação cruzada ' + nome + ' :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    melhor_modelo.fit(X_treino, Y_treino)
    preds = melhor_modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))


def mostrar_features_mais_importantes(melhor_modelo):
    if nome == 'RF':
        print('Características mais importantes RF :')
        feature_importances = pd.DataFrame(melhor_modelo.feature_importances_,
                                           index=X_treino.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)


def escolher_parametros():
    if nome == 'K-NN':
        return [
            {'n_neighbors': [1, 4, 5, 10, 15],
             'weights': ['uniform', 'distance']}
        ]
    elif nome == 'SVM':
        return [
            {'kernel': ['rbf'],
             'gamma': [0.01, 0.1, 1, 5],
             'C': [1, 500, 1000]
             }
            # {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            # 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # },
            # {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            # }
        ]
    elif nome == 'DTC':
        return [
            #{'max_features': [1, 10, 13, 20, 27],
             #'max_depth': [1, 10, 15, 16, 17],
             #'min_samples_split': range(10, 100, 5),
             #'min_samples_leaf': range(1, 30, 2),
             #'class_weight': [None, 'balanced']
             #}
            {'max_features': range(20, 27, 1),
             'max_depth': [13, 14],
             'min_samples_split': range(5, 10, 1),
             'min_samples_leaf': range(15, 20, 1),
             #'class_weight': [None, 'balanced']
             }
        ]
    elif nome == 'NB':
        return [
            {'alpha': range(5, 10, 1),
             'fit_prior': [True, False],
             'class_prior': [None, [1, 2, 3, 4, 5]]}
        ]
    elif nome == 'RF':
        return [
            {'n_estimators': range(10, 300, 50), 'max_features': [10, 20, 27]},
            # {'bootstrap': [False], 'n_estimators': [10, 50, 70], 'max_features': [10, 20, 27]}
        ]
    return None


def imprimir_resultados():
    resultado = pd.DataFrame()
    resultado["id"] = X_teste.index
    resultado["item.acabamento"] = preds
    resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)


for nome, modelo in modelos_base:
    rodar_algoritmos()
    imprimir_resultados()
