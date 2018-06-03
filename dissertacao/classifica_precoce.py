# coding=utf-8
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)


def detect_outliers(series, whis=1.5):
    q75, q25 = np.percentile(series, [75, 25])
    iqr = q75 - q25
    return ~((series - series.median()).abs() <= (whis * iqr))


dados = pd.read_csv('../input/campos_precoce_dados_teste.csv')

# categorizando as colunas
dados.set_index('id', inplace=True)
dados["estabelecimento.cidade"] = dados["estabelecimento.cidade"].astype('object')
dados["item.sexo"] = dados["item.sexo"].astype('category')
dados["game_event_id"] = dados["game_event_id"].astype('category')
dados["game_id"] = dados["game_id"].astype('category')
dados["period"] = dados["period"].astype('object')
dados["playoffs"] = dados["playoffs"].astype('category')
dados["season"] = dados["season"].astype('category')
dados["shot_made_flag"] = dados["shot_made_flag"].astype('category')
dados["shot_type"] = dados["shot_type"].astype('category')
dados["team_id"] = dados["team_id"].astype('category')

dados_copia = dados.copy()
Y_copia = dados_copia['shot_made_flag'].copy()

# removendo colunas irrelevantes
dados_copia.drop('estabelecimento.uf', axis=1, inplace=True)  # sempre igual
dados_copia.drop('item.peso', axis=1, inplace=True)  # o peso depois do abate não é relevante
dados_copia.drop('item.acabamento', axis=1, inplace=True)  # coluna Y

# transformando em dados
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

X = dados_copia[~resultado_desconhecido]
Y = Y_copia[~resultado_desconhecido]

# mostrando o tamanho dos conjuntos
print('Dataset limpo: {}'.format(dados_copia.shape))
print('Dataset para submissão: {}'.format(dados_envio.shape))
print('Dataset de treino: {}'.format(X.shape))
print('Dataset das classes: {}'.format(Y.shape))

seed = 7
num_folds = 7
processors = 1
scoring = 'roc_auc'
kfold = KFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos = [('SVM', LinearSVC(C=1.0)),
           ('K-NN', KNeighborsClassifier(n_neighbors=20)),
           ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)),
           ('NB', MultinomialNB()),
           ('RNA', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))]

# Validar cada um dos modelos
for nome, modelo in modelos:
    cv_resultados = cross_val_score(modelo, X, Y, scoring=scoring, cv=kfold, n_jobs=1)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    modelo.fit(X, Y)
    preds = modelo.predict(dados_envio)
    submission = pd.DataFrame()
    submission["shot_id"] = dados_envio.index
    submission["shot_made_flag"] = preds
    submission.to_csv("sub_" + nome + ".csv", index=False)
