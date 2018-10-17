import warnings
import time
import pydotplus
import graphviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, StratifiedKFold, \
    StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


# mostrar_quantidade_por_classe(1)
# mostrar_quantidade_por_classe(2)
# mostrar_quantidade_por_classe(3)
# mostrar_quantidade_por_classe(4)
# mostrar_quantidade_por_classe(5)


def buscar_quantidades_iguais(quantidade, classe):
    classe = dados_completo.loc[dados_completo['acabamento'] == classe]
    return classe.sample(quantidade, random_state=7)


def mostrar_correlacao(dados, classe):
    matriz_correlacao = dados.corr()
    print('Correlaçao com ' + classe + '\n', matriz_correlacao[classe].sort_values(ascending=False))


mostrar_correlacao(dados_completo, 'acabamento')

# classe_1 = buscar_quantidades_iguais(199, 1)
# classe_2 = buscar_quantidades_iguais(199, 2)
# classe_3 = buscar_quantidades_iguais(199, 3)
# classe_4 = buscar_quantidades_iguais(199, 4)
# classe_5 = buscar_quantidades_iguais(198, 5)
# dados_qtde_iguais = classe_2.append(classe_3).append(classe_4).append(classe_1).append(classe_5)
# dados_qtde_iguais.to_csv("DadosBalanceados.csv", encoding='utf-8', index=False)

# conjunto_treinamento, conjunto_teste = train_test_split(dados_completo, test_size=0.2, random_state=7)
conjunto_treinamento = pd.DataFrame()
conjunto_teste = pd.DataFrame()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
for trainamento_index, teste_index in split.split(dados_completo, dados_completo['acabamento']):
    conjunto_treinamento = dados_completo.loc[trainamento_index]
    conjunto_teste = dados_completo.loc[teste_index]

# X_treino, Y_treino = dados_qtde_iguais.drop('acabamento', axis=1), dados_qtde_iguais['acabamento']
X_treino, X_teste, Y_treino, Y_teste = conjunto_treinamento.drop('acabamento', axis=1), conjunto_teste.drop(
    'acabamento', axis=1), conjunto_treinamento['acabamento'], conjunto_teste['acabamento']
print('X Treino:', X_treino.info())
print('X Teste:', X_teste.info())
mostrar_quantidade_por_classe(conjunto_treinamento, 1)
mostrar_quantidade_por_classe(conjunto_treinamento, 2)
mostrar_quantidade_por_classe(conjunto_treinamento, 3)
mostrar_quantidade_por_classe(conjunto_treinamento, 4)
mostrar_quantidade_por_classe(conjunto_treinamento, 5)
# resultado = pd.DataFrame()
# resultado["id"] = Y_teste.index
# resultado["item.acabamento"] = Y_teste.values
# resultado.to_csv("y_teste.csv", encoding='utf-8', index=False)

seed = 7
num_folds = 5
scoring = 'accuracy'
# scoring = 'precision_macro'
kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos_base = [('NB', MultinomialNB()),
                ('DTC', tree.DecisionTreeClassifier()),
                ('K-NN', KNeighborsClassifier()),  # n_jobs=-1 roda com o mesmo número de cores
                ('SVM', SVC())]


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def rodar_svm():
    params = [
        {'kernel': ['rbf'],
         'gamma': [0.01, 0.1, 1, 5],
         'C': [1, 500, 1000]
         }]
    inicio = time.time()
    #    grid_search = GridSearchCV(SVC(), params, cv=kfold, n_jobs=-1, scoring=scoring)
    #    grid_search.fit(X_treino, Y_treino)
    #    melhor_modelo = grid_search.best_estimator_
    melhor_modelo = BaggingClassifier(SVC(kernel='rbf', gamma=5, C=1000), max_samples=1/num_folds)
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    #    print('Melhores parametros SVM :', grid_search.best_estimator_)
    print('Validação cruzada SVM :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format('SVM', cv_resultados.mean(), cv_resultados.std()))
    # melhor_modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do SVM: {0:.4f} segundos'.format(final - inicio))
    # fig, sub = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # x0, x1 = X_treino['maturidade'], X_treino['peso_carcaca']
    # xx, yy = make_meshgrid(x0, x1)

    # plot_contours(sub.flatten(), melhor_modelo, xx, yy,
    #               cmap=plt.cm.coolwarm, alpha=0.8)
    # sub.flatten().scatter(x0, x1, c=Y_treino, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    # sub.flatten().set_xlim(xx.min(), xx.max())
    # sub.flatten().set_ylim(yy.min(), yy.max())
    # sub.flatten().set_xlabel('Maturidade')
    # sub.flatten().set_ylabel('Peso da carcaça')
    # sub.flatten().set_xticks(())
    # sub.flatten().set_yticks(())
    # sub.flatten().set_title('SVM')

    # plt.show()


def rodar_dtc():
    params = [
        {'max_features': [1, 10, 13, 20, 27],
         'max_depth': [1, 10, 15, 16, 17],
         'min_samples_split': range(10, 100, 5),
         'min_samples_leaf': range(1, 30, 2),
         'class_weight': [None, 'balanced']
         }
    ]
    inicio = time.time()
    grid_search = GridSearchCV(tree.DecisionTreeClassifier(), params, cv=kfold, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    # melhor_modelo = tree.DecisionTreeClassifier(max_features=, max_depth=, class_weight=)
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros DTC :', melhor_modelo)
    print('Melhor score DTC :', grid_search.best_score_)
    print('Validação cruzada DTC :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format('DTC', cv_resultados.mean(), cv_resultados.std()))
    # melhor_modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do DTC: {0:.4f} segundos'.format(final - inicio))


#    dot_data = StringIO()
#    export_graphviz(melhor_modelo, out_file=dot_data,
#                    filled=True, rounded=True,
#                    special_characters=True)
#    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#    Image(graph.create_png())
#    dot_data = tree.export_graphviz(melhor_modelo, out_file=None)
#    graph = graphviz.Source(dot_data)
#    graph.render("Precoce MS")
#    dot_data = tree.export_graphviz(melhor_modelo, out_file=None,
#                                    feature_names=conjunto_treinamento.feature_names,
#                                    class_names=conjunto_treinamento.target_names,
#                                    filled=True, rounded=True,
#                                    special_characters=True)
#    graph = graphviz.Source(dot_data)
#    graph.show()


def rodar_nb():
    # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 11, 12]
    # params = {'alpha': alphas, 'fit_prior': [True, False], 'class_prior': [None, [1, 2, 3, 4, 5]]}
    inicio = time.time()
    # grid_search = GridSearchCV(MultinomialNB(), params, cv=kfold, n_jobs=-1)
    # grid_search.fit(X_treino, Y_treino)
    # melhor_modelo = grid_search.best_estimator_
    melhor_modelo = MultinomialNB(alpha=11)
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros NB :', melhor_modelo)
    # print('Melhor score NB :', grid_search.best_score_)
    print('Validação cruzada NB :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format('NB', cv_resultados.mean(), cv_resultados.std()))
    # melhor_modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do NB: {0:.4f} segundos'.format(final - inicio))


def rodar_knn():
    params = [
        {'n_neighbors': [1, 4, 5, 10, 15],
         'weights': ['uniform', 'distance']}
    ]
    inicio = time.time()
    grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=kfold, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros K-NN :', melhor_modelo)
    print('Melhor score K-NN :', grid_search.best_score_)
    print('Validação cruzada K-NN :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format('K-NN', cv_resultados.mean(), cv_resultados.std()))
    # melhor_modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do K-NN: {0:.4f} segundos'.format(final - inicio))


def rodar_rf():
    params = [
        {'n_estimators': [10, 50, 70], 'max_features': [10, 20, 26, 27]},
        # {'bootstrap': [False], 'n_estimators': [10, 50, 70], 'max_features': [10, 20, 27]}
    ]
    inicio = time.time()
    grid_search = GridSearchCV(RandomForestClassifier(random_state=seed), params, cv=kfold, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_treino, Y_treino)
    melhor_modelo = grid_search.best_estimator_
    cv_resultados = cross_val_score(melhor_modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print('Melhores parametros RF :', melhor_modelo)
    print('Melhor score RF :', grid_search.best_score_)
    print('Características mais importantes RF :')
    feature_importances = pd.DataFrame(melhor_modelo.feature_importances_,
                                       index=X_treino.columns,
                                       columns=(['importance']).sort_values('importance', ascending=False))
    print(feature_importances)
    # feature_importances = grid_search.best_estimator_.feature_importances_
    # indices = np.argsort(feature_importances)[::-1]
    # for f in range(X_treino.shape[1]):
    #     print('%d. feature %d (%f)' % (f + 1, indices[f], feature_importances[indices[f]]))
    print('Validação cruzada RF :', cv_resultados)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format('RF', cv_resultados.mean(), cv_resultados.std()))
    # melhor_modelo.fit(X_treino, Y_treino)
    # preds = modelo.predict(X_teste)
    final = time.time()
    print('Tempo de execução do RF: {0:.4f} segundos'.format(final - inicio))


# def rodar_algoritmos():
#     inicio = time.time()
#     grid_search = GridSearchCV(modelo, escolher_parametros(), cv=kfold, n_jobs=-1)
#     grid_search.fit(X_treino, Y_treino)
#     cv_resultados = cross_val_score(grid_search.best_estimator_, X_treino, Y_treino, cv=kfold, scoring=scoring)
#     print('Melhores parametros ' + nome + ' :', grid_search.best_estimator_)
#     print('Validação cruzada ' + nome + ' :', cv_resultados)
#     print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
#     grid_search.best_estimator_.fit(X_treino, Y_treino)
#     # preds = modelo.predict(X_teste)
#     final = time.time()
#     print('Tempo de execução do ' + nome + ': {0:.4f} segundos'.format(final - inicio))


# def escolher_parametros():
#     if nome == 'K-NN':
#         return [
#             {'n_neighbors': [1, 4, 5, 10, 15],
#              'weights': ['uniform', 'distance']}
#         ]
#     elif nome == 'SVM':
#         return [
#             {'kernel': ['rbf'],
#              'gamma': [0.01, 0.1, 1],
#              'C': [500, 1000, 1500]
#              },
#             # {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#             # 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
#             # },
#             # {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
#             # }
#         ]
#     elif nome == 'DTC':
#         return [
#             {'max_features': [1, 5, 8, 9, 10, 12, 20],
#              'max_depth': [1, 5, 10, 11, 12, 13, 14],
#              'class_weight': [None, 'balanced']
#              }
#         ]
#     elif nome == 'NB':
#         return [
#             {'alpha': [0, .0001, .001, .01, .1, .5, 1, 5, 9, 10, 11, 15]}
#         ]
#     return None


# def imprimir_resultados():
# resultado = pd.DataFrame()
# resultado["id"] = X_teste.index
# resultado["item.acabamento"] = preds
# resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)

rodar_nb()
rodar_dtc()
rodar_rf()

rodar_svm()
rodar_knn()
# Validar cada um dos modelos
# for nome, modelo in modelos_base:
# rodar_algoritmos()
# imprimir_resultados()
