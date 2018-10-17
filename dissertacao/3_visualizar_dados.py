import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)  # display all columns

dados_completo = pd.read_csv('../input/DadosCompletoTransformado.csv', encoding='utf-8', delimiter='\t')
dados_completo.set_index('index', inplace=True)

print(dados_completo.shape)
print(dados_completo.describe(include=['number']))
print(dados_completo.describe(include=['object', 'category']))

train_set, test_set = train_test_split(dados_completo, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")

ax = plt.axes()
sns.countplot(x='acabamento', data=dados_completo, ax=ax)
ax.set_title('Distribuição do acabamento')
plt.savefig('distribuicao_acabamento.svg')
plt.show()

f, axarr = plt.subplots(4, 2, figsize=(15, 15))

sns.boxplot(x='maturidade', y='acabamento', data=dados_completo, showmeans=True, ax=axarr[0, 0])
sns.boxplot(x='rastreamento_sisbov', y='acabamento', data=dados_completo, showmeans=True, ax=axarr[0, 1])
sns.boxplot(x='questionario_classificacao_estabelecimento_rural', y='acabamento', data=dados_completo, showmeans=True,
            ax=axarr[1, 0])
sns.boxplot(x='possui_outros_incentivos', y='acabamento', data=dados_completo, showmeans=True, ax=axarr[1, 1])
sns.boxplot(x='fabrica_racao', y='acabamento', showmeans=True, data=dados_completo, ax=axarr[2, 0])
sns.boxplot(x='area_total_destinada_confinamento', y='acabamento', showmeans=True, data=dados_completo, ax=axarr[2, 1])
sns.boxplot(x='area_manejada_80_boa_cobertura_vegetal', y='acabamento', data=dados_completo, showmeans=True,
            ax=axarr[3, 0])
sns.boxplot(x='tipificacao', y='acabamento', data=dados_completo, showmeans=True,
            ax=axarr[3, 1])

axarr[0, 0].set_title('Maturidade')
axarr[0, 1].set_title('Ratreamento SISBOV?')
axarr[1, 0].set_title('Classificação do estabelecimento rural')
axarr[1, 1].set_title('Possui outros incentivos?')
axarr[2, 0].set_title('Fabrica ração?')
axarr[2, 1].set_title('Área total destinada a confinamento?')
axarr[3, 0].set_title('Área manejada possui 80% de boa cobertura vegetal')
axarr[3, 1].set_title('Tipificação')

plt.tight_layout()
plt.savefig('boxplot.svg')
plt.show()

dados_completo.hist(bins=50, figsize=(25, 25))
plt.savefig('hitograma.svg')
plt.show()

ms_mapa = mpimg.imread('mato-grosso-so-sul.png')
ax = dados_completo.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=dados_completo['peso_carcaca'],
                         label='Peso da Carcaça', c='acabamento', cmap=plt.get_cmap('jet'), colorbar=False,
                         figsize=(10, 7))
plt.imshow(ms_mapa, extent=[-57.88, -51.09, -25.97, -18.57], alpha=0.5, cmap=plt.get_cmap('jet'))
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
acabamentos = dados_completo['acabamento']
tick_values = np.linspace(acabamentos.min(), acabamentos.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(['%d' % (round(v)) for v in tick_values], fontsize=14)
cbar.set_label('Acabamento', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('scatter.png')
print('Foi!!')

atributos = ['acabamento', 'peso_carcaca', 'tipificacao', 'maturidade']
scatter_matrix(dados_completo[atributos], figsize=(12, 8))
plt.savefig('scatter_atributos.png')

# procurando por correlações
matriz_correlacao = dados_completo.corr()
print('Correlação de Pearson:')
print(matriz_correlacao['acabamento'].sort_values(ascending=False))
