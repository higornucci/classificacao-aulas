import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.features import RFECV

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns

num_folds = 5
random_state = 42
scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=num_folds)


def mostrar_quantidade_por_classe(df, classe):
    print(classe)
    print(df.loc[df['classe'] == classe].info())


dados_completo = pd.read_csv('../input/trainingM.csv', encoding='utf-8', delimiter=',')
# dados_completo.drop('index', axis=1, inplace=True)

mostrar_quantidade_por_classe(dados_completo, 'dirtiness')
mostrar_quantidade_por_classe(dados_completo, 'white_bgd')
mostrar_quantidade_por_classe(dados_completo, 'viable')
mostrar_quantidade_por_classe(dados_completo, 'not_viable')
# mostrar_quantidade_por_classe(dados_completo, 5)

print(dados_completo.shape)
print(dados_completo.describe(include=['number']))

n_jobs = 5
# classes_balancear = list([2, 3])
balanceador = EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=5)
# balanceador = SMOTE(n_jobs=n_jobs, random_state=random_state)
# balanceador = SMOTEENN(enn=EditedNearestNeighbours(n_jobs=n_jobs, n_neighbors=n_jobs), smote=SMOTE(n_jobs=n_jobs),
#                        random_state=random_state)

X_treino, Y_treino = balanceador.fit_resample(
    dados_completo.drop('classe', axis=1),
    dados_completo['classe'])
X_treino = pd.DataFrame(data=X_treino, columns=dados_completo.drop(['classe'], axis=1).columns)
Y_treino = pd.DataFrame(data=Y_treino, columns=['classe'])
# X_treino.to_csv('../input/DadosCompletoTransformadoMLBalanceadoX.csv', encoding='utf-8', sep='\t')
# Y_treino.to_csv('../input/DadosCompletoTransformadoMLBalanceadoY.csv', encoding='utf-8', sep='\t')
# # exit()
# X_treino = pd.read_csv('../input/DadosCompletoTransformadoMLBalanceadoX.csv', encoding='utf-8', delimiter='\t')
# X_treino.drop(X_treino.columns[0], axis=1, inplace=True)
# Y_treino = pd.read_csv('../input/DadosCompletoTransformadoMLBalanceadoY.csv', encoding='utf-8', delimiter='\t')
# Y_treino.drop(Y_treino.columns[0], axis=1, inplace=True)
conjunto_balanceado = X_treino.join(Y_treino)
conjunto_balanceado = conjunto_balanceado.sample(frac=1)
print(conjunto_balanceado.shape)
print(conjunto_balanceado.describe())
mostrar_quantidade_por_classe(conjunto_balanceado, 'dirtiness')
mostrar_quantidade_por_classe(conjunto_balanceado, 'white_bgd')
mostrar_quantidade_por_classe(conjunto_balanceado, 'viable')
mostrar_quantidade_por_classe(conjunto_balanceado, 'not_viable')
# mostrar_quantidade_por_classe(conjunto_balanceado, 5)

ax = plt.axes()
sns.countplot(x='classe', data=dados_completo, ax=ax, order=dados_completo['classe'].value_counts().index)
# ax.set_title('Distribution of carcass fatness degree')
plt.savefig('figuras/distribution_imbalanced_seeds.png')
plt.show()

ax = plt.axes()
sns.countplot(x='classe', data=conjunto_balanceado, ax=ax, order=dados_completo['classe'].value_counts().index)
# ax.set_title('Distribution of carcass fatness degree')
plt.savefig('figuras/distribution_balanced_seeds_SMOTEENN.png')
plt.show()


def mostrar_features_mais_importantes(melhor_modelo):
    melhor_modelo.fit(dados_completo.drop(['classe'], axis=1),
                      dados_completo['classe'].values)
    print('Características mais importantes RFC :')
    feature_importances = pd.DataFrame(melhor_modelo.feature_importances_,
                                       index=dados_completo.drop(['classe'], axis=1).columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)


def fazer_selecao_features_rfe(modelo):
    features = dados_completo.columns
    rfe = RFECV(modelo,
                cv=kfold, scoring=scoring)

    rfe.fit(dados_completo.drop(['classe'], axis=1), dados_completo['classe'].values)
    print(rfe.poof())
    print("Caraterísticas ordenadas pelo rank RFE:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))
    ranking = sorted(zip(rfe.support_, features))
    print("Características selecionadas", ranking)
    return rfe.transform(dados_completo.drop(['classe'], axis=1))


modelo = RandomForestClassifier(class_weight='balanced', max_depth=75,
                                max_features='log2', min_samples_leaf=1, min_samples_split=10, n_estimators=100)
mostrar_features_mais_importantes(modelo)
print(fazer_selecao_features_rfe(modelo))
# ax = plt.axes()
# sns.countplot(x='carcass_fatness_degree', data=dados_completo, ax=ax)
# ax.set_title('Distribution of carcass fatness degree')
# plt.savefig('figuras/distribuicao_acabamento.png')
# plt.show()

dados_completo.hist(bins=50, figsize=(50, 50))
plt.savefig('figuras/histograma_seeds.png')
plt.show()

# ms_mapa = mpimg.imread('figuras/mato-grosso-so-sul.png')
#
# ax = dados_completo.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3, s=dados_completo['carcass_weight'],
#                          label='carcass_weight', c='carcass_fatness_degree', cmap=plt.get_cmap('jet'), colorbar=True,
#                          figsize=(10, 7))
# plt.imshow(ms_mapa, extent=[-58.88, -50.2, -24.8, -16.57], alpha=0.5, cmap=plt.get_cmap('jet'))
# plt.legend(fontsize=12)
# plt.savefig('figuras/scatter_ms.png')
# print('Foi!!')

# procurando por correlações
matriz_correlacao = dados_completo.corr()
print('Correlação de Pearson:')
print(matriz_correlacao['classe'].sort_values(ascending=False))

# sns.pairplot(dados_completo, kind="scatter", hue="carcass_fatness_degree", markers=["o", "s", "D"], palette="Set2")
# sns.pairplot(dados_completo, kind="scatter", hue="carcass_fatness_degree", palette="Set2")
# plt.savefig('figuras/matriz_correlacao_pares.png')
# plt.show()
#
# sns.pairplot(dados_completo, kind="reg")
# plt.savefig('figuras/scatter_atributos.png')
# plt.show()
