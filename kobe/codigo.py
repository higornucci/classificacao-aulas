# coding=utf-8

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree
from sklearn import svm
from random import randint

import numpy as np
import pandas as pd
import csv

dados = pd.read_csv('data.csv')

print 'Tratando dados'
# categorizando as colunas
dados.set_index('shot_id', inplace=True)
dados["action_type"] = dados["action_type"].astype('object')
dados["combined_shot_type"] = dados["combined_shot_type"].astype('category')
dados["game_event_id"] = dados["game_event_id"].astype('category')
dados["game_id"] = dados["game_id"].astype('category')
dados["period"] = dados["period"].astype('object')
dados["playoffs"] = dados["playoffs"].astype('category')
dados["season"] = dados["season"].astype('category')
dados["shot_made_flag"] = dados["shot_made_flag"].astype('category')
dados["shot_type"] = dados["shot_type"].astype('category')
dados["team_id"] = dados["team_id"].astype('category')

resultado_desconhecido = dados['shot_made_flag'].isnull()
dados_copia = dados.copy()
Y_copia = dados_copia['shot_made_flag'].copy()

# removendo colunas irrelevantes
dados_copia.drop('team_id', axis=1, inplace=True)  # sempre igual
dados_copia.drop('lat', axis=1, inplace=True)  # mesma que loc_x
dados_copia.drop('lon', axis=1, inplace=True)  # mesma que loc_y
dados_copia.drop('game_id', axis=1, inplace=True)  # não relevante
dados_copia.drop('game_event_id', axis=1, inplace=True)  # não relevante
dados_copia.drop('team_name', axis=1, inplace=True)  # sempre o mesmo
dados_copia.drop('shot_made_flag', axis=1, inplace=True)  # coluna Y

# transformando os dados
# tempo restante
dados_copia['seconds_from_period_end'] = 60 * dados_copia['minutes_remaining'] + dados_copia['seconds_remaining']
dados_copia['last_5_sec_in_period'] = dados_copia['seconds_from_period_end'] < 5

dados_copia.drop('minutes_remaining', axis=1, inplace=True)
dados_copia.drop('seconds_remaining', axis=1, inplace=True)
dados_copia.drop('seconds_from_period_end', axis=1, inplace=True)

# Partida - (fora/em casa)
dados_copia['home_play'] = dados_copia['matchup'].str.contains('vs').astype('int')
dados_copia.drop('matchup', axis=1, inplace=True)

# dia do jogo
dados_copia['game_date'] = pd.to_datetime(dados_copia['game_date'])
dados_copia['game_year'] = dados_copia['game_date'].dt.year
dados_copia['game_month'] = dados_copia['game_date'].dt.month
dados_copia.drop('game_date', axis=1, inplace=True)

# localização
dados_copia['loc_x'] = pd.cut(dados_copia['loc_x'], 25)
dados_copia['loc_y'] = pd.cut(dados_copia['loc_y'], 25)

# trocar os valores comuns para other
rare_action_types = dados_copia['action_type'].value_counts().sort_values().index.values[:20]
dados_copia.loc[dados_copia['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

colunas_categoricas = [
    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'loc_x', 'loc_y']

for cc in colunas_categoricas:
    dummies = pd.get_dummies(dados_copia[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    dados_copia.drop(cc, axis=1, inplace=True)
    dados_copia = dados_copia.join(dummies)

dados_envio = dados_copia[resultado_desconhecido]

print 'Dados já foram tratados'
print 'Separando dados para treino e teste'

X = dados_copia[~resultado_desconhecido]
Y = Y_copia[~resultado_desconhecido]

print 'Dados já foram separados'
print 'Iniciando algoritmos'

predicao = list()
quantidade = 100

arquivo2 = csv.writer(open("resultados.csv", "wb"))
arquivo2.writerow(['Algoritmo', 'score'])

seed = 7
num_folds = 7
processors = 1
scoring = 'roc_auc'
kfold = KFold(n_splits=num_folds, random_state=seed)

# ---------- KNN ----------
print 'Executando KNN'
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [30, 35, 40, 45, 50]
p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

arquivo = csv.writer(open("KNN.csv", "wb"))
arquivo.writerow(['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'score'])

listaKNN = list()
listaKNN1 = list()

z = 0
for i in range(quantidade):
    j = n_neighbors[randint(0, 9)]
    k = weights[randint(0, 1)]
    l = algorithm[randint(0, 3)]
    m = leaf_size[randint(0, 4)]
    n = p[randint(0, 9)]

    clf = KNeighborsClassifier(n_neighbors=j, weights=k, algorithm=l, leaf_size=m, p=n, metric='minkowski')

    cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([j, k, l, m, n, score])
    print z
    z = z + 1
    listaKNN.append(score)
    listaKNN1.append([j, k, l, m, n])

maior = listaKNN.index(max(listaKNN))
predicao = listaKNN1[maior]

clf = KNeighborsClassifier(n_neighbors=predicao[0], weights=predicao[1], algorithm=predicao[2], leaf_size=predicao[3],
                           p=predicao[4], metric='minkowski')
clf = clf.fit(X, Y)
score = max(listaKNN)
arquivo2.writerow(['KNN', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_KNN.csv", index=False)

print "Concluido KNN. Maior score: "
print max(listaKNN)
del (listaKNN)
del (listaKNN1)
print '----------'

# ---------- SVM SVC ----------

print 'Executando SVM SVC'
tol = [0.1, 0.01, 0.001, 0.0001, 0.00001]
C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
multi_class = ['ovr', 'crammer_singer']
fit_intercept = ['False', 'True']
intercept_scaling = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
verbose = [0, 1, 2, 3, 4, 5]
random_state = [1, 2, 3, 4, 5]
max_iter = [500, 1000, 1500]

listaSVM = list()
listaSVM1 = list()
arquivo = csv.writer(open("SVM.csv", "wb"))
arquivo.writerow(
    ['tol', 'C', 'multi_class', 'fit_intercept', 'intercept_scaling', 'verbose', 'random_state', 'max_iter', 'score'])
z = 0
for i in range(quantidade):
    m = tol[randint(0, 4)]
    n = C[randint(0, 9)]
    o = multi_class[randint(0, 1)]
    p = fit_intercept[randint(0, 1)]
    q = intercept_scaling[randint(0, 9)]
    r = verbose[randint(0, 5)]
    s = random_state[randint(0, 4)]
    t = max_iter[randint(0, 2)]

    clf = svm.LinearSVC(tol=m, C=n, multi_class=o, fit_intercept=p, intercept_scaling=q, class_weight='balanced',
                        verbose=r, random_state=s, max_iter=t)

    cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([m, n, o, p, q, r, s, t, score])
    print z
    z = z + 1
    listaSVM.append(score)
    listaSVM1.append([m, n, o, p, q, r, s, t])

maior = listaSVM.index(max(listaSVM))
predicao = listaSVM1[maior]

clf = svm.LinearSVC(tol=predicao[0], C=predicao[1], multi_class=predicao[2], fit_intercept=predicao[3],
                    intercept_scaling=predicao[4], class_weight='balanced', verbose=predicao[5],
                    random_state=predicao[6], max_iter=predicao[7])
clf = clf.fit(X, Y)
score = max(listaSVM)
arquivo2.writerow(['SVM', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_SVM.csv", index=False)

print "Concluido SVM. Maior score: "
print max(listaSVM)
del (listaSVM)
del (listaSVM1)
print '----------'

# ---------- Naive Bayes (GaussianNB) ----------
print 'Executando Naive Bayes (GaussianNB)'
arquivo = csv.writer(open("GaussianNB.csv", "wb"))
arquivo.writerow(['score'])

clf = GaussianNB()

cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
score = cv_resultados.mean()

clf = clf.fit(X, Y)
arquivo.writerow([score])
arquivo2.writerow(['GaussianNB', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_GaussianNB.csv", index=False)

print "Concluido GaussianNB. Score: "
print score
print '----------'

# ---------- Naive Bayes (MultinomialNB) ----------
print 'Executando Naive Bayes (MultinomialNB)'
arquivo = csv.writer(open("MultinomialNB.csv", "wb"))
arquivo.writerow(['alpha', 'fit_prior', 'score'])

alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fit_prior = ['False', 'True']

listaMultinomialNB = list()
listaMultinomialNB1 = list()
z = 0
for i in range(quantidade):
    j = alpha[randint(0, 10)]
    k = fit_prior[randint(0, 1)]

    clf = MultinomialNB(alpha=j, fit_prior=k, class_prior=None)

    cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([j, k, score])
    print z
    z = z + 1
    listaMultinomialNB.append(score)
    listaMultinomialNB1.append([j, k])

maior = listaMultinomialNB.index(max(listaMultinomialNB))
predicao = listaMultinomialNB1[maior]

clf = MultinomialNB(alpha=predicao[0], fit_prior=predicao[1], class_prior=None)
clf = clf.fit(X, Y)
score = max(listaMultinomialNB)
arquivo2.writerow(['MultinomialNB', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_MultinomialNB.csv", index=False)

print "Concluido MultinomialNB. Maior score: "
print max(listaMultinomialNB)
del (listaMultinomialNB)
del (listaMultinomialNB1)
print '----------'

# ---------- Arvore de Decisao ----------
print 'Executando Arvore de Decisao'
arquivo = csv.writer(open("arvore.csv", "wb"))
arquivo.writerow(
    ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'score'])

criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

listaArvore = list()
listaArvore1 = list()
z = 0
for i in range(quantidade):
    j = criterion[randint(0, 1)]
    k = splitter[randint(0, 1)]
    l = max_depth[randint(0, 9)]
    m = min_samples_split[randint(0, 8)]
    n = min_samples_leaf[randint(0, 9)]
    o = max_features[randint(0, 9)]

    clf = tree.DecisionTreeClassifier(criterion=j, splitter=k, max_depth=l, min_samples_split=m, min_samples_leaf=n,
                                      min_weight_fraction_leaf=0, max_features=o, random_state=None,
                                      max_leaf_nodes=None, min_impurity_decrease=0, class_weight='balanced',
                                      presort=False)

    cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([j, k, l, m, n, o, score])
    print z
    z = z + 1
    listaArvore.append(score)
    listaArvore1.append([j, k, l, m, n, o])

maior = listaArvore.index(max(listaArvore))
predicao = listaArvore1[maior]

clf = tree.DecisionTreeClassifier(criterion=predicao[0], splitter=predicao[1], max_depth=predicao[2],
                                  min_samples_split=predicao[3], min_samples_leaf=predicao[4],
                                  min_weight_fraction_leaf=0, max_features=predicao[5], random_state=None,
                                  max_leaf_nodes=None, min_impurity_decrease=0, class_weight='balanced', presort=False)
clf = clf.fit(X, Y)
score = max(listaArvore)
arquivo2.writerow(['Arvore', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_Arvore.csv", index=False)

print "Concluido Arvore de Decisao. Maior score: "
print max(listaArvore)
del (listaArvore)
del (listaArvore1)
print '----------'

# ---------- Rede Neural (dados sem normalizar) ----------
print 'Executando Rede Neural (dados sem normalizar)'
arquivo = csv.writer(open("rede_naoNormalizado.csv", "wb"))
arquivo.writerow(
    ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', 'learning_rate_init', 'power_t',
     'random_state', 'tol', 'verbose', 'warm_start', 'momentum', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon',
     'score'])

# random search
hidden_layer_sizes = [50, 100, 150]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
alpha = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_init = [1.0, 0.1, 0.01, 0.001]
power_t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
random_state = [0, 1, 2, 3, 4, 5]
tol = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
verbose = ['False', 'True']
warm_start = ['False', 'True']
momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
validation_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]

listaRede = list()
listaRede1 = list()
z = 0
for i in range(quantidade):
    j = hidden_layer_sizes[randint(0, 2)]
    k = activation[randint(0, 3)]
    l = solver[randint(0, 2)]
    m = alpha[randint(0, 6)]
    n = learning_rate[randint(0, 2)]
    o = learning_rate_init[randint(0, 3)]
    p = power_t[randint(0, 5)]
    q = random_state[randint(0, 5)]
    r = tol[randint(0, 5)]
    s = verbose[randint(0, 1)]
    t = warm_start[randint(0, 1)]
    u = momentum[randint(0, 10)]
    v = validation_fraction[randint(0, 8)]
    w = beta_1[randint(0, 9)]
    x = beta_2[randint(0, 9)]
    y = epsilon[randint(0, 8)]

    clf = MLPClassifier(hidden_layer_sizes=j, activation=k, solver=l, alpha=m, learning_rate=n, learning_rate_init=o,
                        power_t=p, random_state=q, tol=r, verbose=s, warm_start=t, momentum=u, validation_fraction=v,
                        beta_1=w, beta_2=x, epsilon=y)

    cv_resultados = cross_val_score(clf, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, score])
    print z
    z = z + 1
    listaRede.append(score)
    listaRede1.append([j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y])

maior = listaRede.index(max(listaRede))
predicao = listaRede1[maior]

clf = MLPClassifier(hidden_layer_sizes=predicao[0], activation=predicao[1], solver=predicao[2], alpha=predicao[3],
                    learning_rate=predicao[4], learning_rate_init=predicao[5], power_t=predicao[6],
                    random_state=predicao[7], tol=predicao[8], verbose=predicao[9], warm_start=predicao[10],
                    momentum=predicao[11], validation_fraction=predicao[12], beta_1=predicao[13], beta_2=predicao[14],
                    epsilon=predicao[15])

clf = clf.fit(X, Y)
score = max(listaRede)
arquivo2.writerow(['RN sem normalizar', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_RedeSemNormalizar.csv", index=False)

print "Concluido Rede Neural (dados sem normalizar). Maior score: "
print max(listaRede)
del (listaRede)
del (listaRede1)
print '----------'

# ---------- Rede Neural (dados normalizados) ----------
print 'Executando Rede Neural (dados normalizados)'
arquivo = csv.writer(open("rede_normalizado.csv", "wb"))
arquivo.writerow(
    ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', 'learning_rate_init', 'power_t',
     'random_state', 'tol', 'verbose', 'warm_start', 'momentum', 'validation_fraction', 'beta_1', 'beta_2', 'epsilon',
     'score'])

# random search
hidden_layer_sizes = [50, 100, 150]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
alpha = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_init = [1.0, 0.1, 0.01, 0.001]
power_t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
random_state = [0, 1, 2, 3, 4, 5]
tol = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
verbose = ['False', 'True']
warm_start = ['False', 'True']
momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
validation_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]

X1 = X
scaler = MinMaxScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)

listaRede2 = list()
listaRede3 = list()
z = 0
for i in range(quantidade):
    j = hidden_layer_sizes[randint(0, 2)]
    k = activation[randint(0, 3)]
    l = solver[randint(0, 2)]
    m = alpha[randint(0, 6)]
    n = learning_rate[randint(0, 2)]
    o = learning_rate_init[randint(0, 3)]
    p = power_t[randint(0, 5)]
    q = random_state[randint(0, 5)]
    r = tol[randint(0, 5)]
    s = verbose[randint(0, 1)]
    t = warm_start[randint(0, 1)]
    u = momentum[randint(0, 10)]
    v = validation_fraction[randint(0, 8)]
    w = beta_1[randint(0, 9)]
    x = beta_2[randint(0, 9)]
    y = epsilon[randint(0, 8)]

    clf = MLPClassifier(hidden_layer_sizes=j, activation=k, solver=l, alpha=m, learning_rate=n, learning_rate_init=o,
                        power_t=p, random_state=q, tol=r, verbose=w, warm_start=t, momentum=u, validation_fraction=v,
                        beta_1=w, beta_2=x, epsilon=y)

    cv_resultados = cross_val_score(clf, X1, Y, scoring=scoring, cv=kfold, n_jobs=1)
    score = cv_resultados.mean()
    arquivo.writerow([j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, score])
    print z
    z = z + 1
    listaRede2.append(score)
    listaRede3.append([j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y])

maior = listaRede2.index(max(listaRede2))
predicao = listaRede3[maior]

clf = MLPClassifier(hidden_layer_sizes=predicao[0], activation=predicao[1], solver=predicao[2], alpha=predicao[3],
                    learning_rate=predicao[4], learning_rate_init=predicao[5], power_t=predicao[6],
                    random_state=predicao[7], tol=predicao[8], verbose=predicao[9], warm_start=predicao[10],
                    momentum=predicao[11], validation_fraction=predicao[12], beta_1=predicao[13], beta_2=predicao[14],
                    epsilon=predicao[15])
clf = clf.fit(X1, Y)
score = max(listaRede2)
arquivo2.writerow(['RN normalizado', score])
predicao = clf.predict(dados_envio)
submission = pd.DataFrame()
submission["shot_id"] = dados_envio.index
submission["shot_made_flag"] = predicao
submission.to_csv("sub_RedeNormalizado.csv", index=False)

print "Concluido Rede Neural (dados normalizados). Maior score: "
print max(listaRede2)
del (listaRede2)
del (listaRede3)
print '----------'
