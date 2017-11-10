import pandas as pd

df = pd.read_csv('../input/data.csv')

X_df = df[['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_date', 'matchup', 'opponent']]
Y_df = df['shot_made_flag']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

tamanho_de_teste = len(Y) - tamanho_de_treino
teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_treino:]
