import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from imblearn.under_sampling import EditedNearestNeighbours
from mpl_toolkits.mplot3d import Axes3D

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', 2000)  # display all columns


def mostrar_quantidade_por_classe(df, classe):
    print(df.loc[df['acabamento'] == classe].info())


dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.drop('index', axis=1, inplace=True)

mostrar_quantidade_por_classe(dados_completo, 1)
mostrar_quantidade_por_classe(dados_completo, 2)
mostrar_quantidade_por_classe(dados_completo, 3)
mostrar_quantidade_por_classe(dados_completo, 4)
mostrar_quantidade_por_classe(dados_completo, 5)

print(dados_completo.shape)
print(dados_completo.describe(include=['number']))

balanceador = EditedNearestNeighbours(n_jobs=-1, sampling_strategy=list([1, 2, 3, 4]))
X_treino, Y_treino = balanceador.fit_resample(
    dados_completo.drop(['acabamento'], axis=1),
    dados_completo['acabamento'])
X_treino = pd.DataFrame(data=X_treino, columns=dados_completo.drop(['acabamento'], axis=1).columns)
Y_treino = pd.DataFrame(data=Y_treino, columns=['acabamento'])
conjunto_balanceado = X_treino.join(Y_treino)
conjunto_balanceado = conjunto_balanceado.sample(frac=1)
print(conjunto_balanceado.shape)
print(conjunto_balanceado.describe(include=['number']))


def plotar_dataset_2d_imbalanced():
    grafico = pd.DataFrame()
    grafico['maturidade'] = dados_completo['maturidade'] + dados_completo['tipificacao']
    grafico['peso_carcaca'] = dados_completo['peso_carcaca']
    grafico['acabamento'] = dados_completo['acabamento']
    sns.lmplot(x="maturidade", y="peso_carcaca", data=grafico, hue="acabamento", fit_reg=False, legend=False)
    plt.legend()
    plt.savefig('dataset_completo2d_imbalanced.png')
    plt.show()


def plotar_dataset_2d_balanced():
    grafico = pd.DataFrame()
    grafico['maturidade'] = conjunto_balanceado['maturidade'] + conjunto_balanceado['tipificacao']
    grafico['peso_carcaca'] = conjunto_balanceado['peso_carcaca']
    grafico['acabamento'] = conjunto_balanceado['acabamento']
    sns.lmplot(x="maturidade", y="peso_carcaca", data=grafico, hue="acabamento", fit_reg=False, legend=False)
    plt.legend()
    plt.savefig('dataset_completo2d_balanced.png')
    plt.show()


plotar_dataset_2d_balanced()
# plotar_dataset_2d_imbalanced()


def plotar_dataset_3d_imbalanced():
    x = pd.DataFrame(np.array(dados_completo['tipificacao']).reshape(-1, 1))
    y = pd.DataFrame(np.array(dados_completo['maturidade']).reshape(-1, 1))
    z = pd.DataFrame(np.array(dados_completo['peso_carcaca']).reshape(-1, 1))
    target = pd.DataFrame(np.array(dados_completo['acabamento']).reshape(-1, 1))

    new_data = [x, y, z, target]

    new_data = pd.concat(new_data, axis=1, ignore_index=True)

    new_data.columns = ['typification', 'maturity', 'carcass_weight', 'carcass_fatness_degree']

    colors = new_data['carcass_fatness_degree'].map(
        {1: 'blue', 2: 'darkorange', 3: 'green', 4: 'crimson', 5: 'purple'})

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(111, projection='3d')
    ax2 = Axes3D(fig)
    ax2.scatter(new_data.maturity, new_data.typification, new_data.carcass_weight, c=colors)
    ax2.set_xlabel('maturity')
    ax2.set_ylabel('typification')
    ax2.set_zlabel('carcass_weight')
    plt.legend()
    plt.savefig('dataset_completo3d_imbalanced.png')
    plt.show()


# def plotar_dataset_3d_balanced():
#     x = pd.DataFrame(np.array(conjunto_balanceado['tipificacao']).reshape(-1, 1))
#     y = pd.DataFrame(np.array(conjunto_balanceado['maturidade']).reshape(-1, 1))
#     z = pd.DataFrame(np.array(conjunto_balanceado['peso_carcaca']).reshape(-1, 1))
#     target = pd.DataFrame(np.array(conjunto_balanceado['acabamento']).reshape(-1, 1))
#
#     new_data = [x, y, z, target]
#
#     new_data = pd.concat(new_data, axis=1, ignore_index=True)
#
#     new_data.columns = ['typification', 'maturity', 'carcass_weight', 'carcass_fatness_degree']
#
#     colors = new_data['carcass_fatness_degree'].map(
#         {1: 'blue', 2: 'darkorange', 3: 'green', 4: 'crimson', 5: 'purple'})
#
#     fig = plt.figure(figsize=(8, 8))
#     fig.add_subplot(111, projection='3d')
#     ax2 = Axes3D(fig)
#     ax2.scatter(new_data.maturity, new_data.typification, new_data.carcass_weight, c=colors)
#     ax2.set_xlabel('maturity')
#     ax2.set_ylabel('typification')
#     ax2.set_zlabel('carcass_weight')
#     ax2.legend()
#     plt.savefig('dataset_completo3d_balanced.png')
#     plt.show()


# plotar_dataset_3d_imbalanced()
# plotar_dataset_3d_balanced()

# ax = plt.axes()
# sns.countplot(x='acabamento', data=dados_completo, ax=ax)
# ax.set_title('Distribution of carcass fatness degree')
# plt.savefig('distribuicao_acabamento_desbalanceada.png')
# plt.show()
#
# ax = plt.axes()
# sns.countplot(x='acabamento', data=conjunto_balanceado, ax=ax)
# ax.set_title('Distribution of carcass fatness degree')
# plt.savefig('distribuicao_acabamento_balanceada.png')
# plt.show()

# dados_completo.hist(bins=50, figsize=(25, 25))
# plt.savefig('histograma.png')
# plt.show()

ms_mapa = mpimg.imread('mato-grosso-so-sul.png')

ax = dados_completo.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3, s=dados_completo['peso_carcaca'],
                         label='carcass_weight', c='acabamento', cmap=plt.get_cmap('jet'), colorbar=True,
                         figsize=(10, 7))
plt.imshow(ms_mapa, extent=[-58.88, -50.2, -24.8, -16.57], alpha=0.5, cmap=plt.get_cmap('jet'))
plt.legend(fontsize=12)
plt.savefig('scatter_ms.png')
print('Foi!!')

# procurando por correlações
matriz_correlacao = dados_completo.corr()
print('Correlação de Pearson:')
print(matriz_correlacao['acabamento'].sort_values(ascending=False))

# # sns.pairplot(dados_completo, kind="scatter", hue="acabamento", markers=["o", "s", "D"], palette="Set2")
# sns.pairplot(dados_completo, kind="scatter", hue="acabamento", palette="Set2")
# plt.savefig('matriz_correlacao_pares.png')
# plt.show()

# sns.pairplot(dados_completo, kind="reg")
# plt.savefig('scatter_atributos.png')
# plt.show()
