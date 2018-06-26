import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

dados_precoce = pd.read_csv('../input/ClassificacaoAnimal.csv', encoding='ISO-8859-1', delimiter='\t')

# Substituindo os valores do acabamento
dados_precoce['Acabamento'] = dados_precoce['Acabamento'].replace(['Gordura Escassa - 1 a 3 mm de espessura'], '2')
dados_precoce['Acabamento'] = dados_precoce['Acabamento'].replace(['Gordura Mediana - acima de 3 a até 6 mm de espessura'], '3')
dados_precoce['Acabamento'] = dados_precoce['Acabamento'].replace(['Gordura Uniforme - acima de 6 e até 10 mm de espessura'], '4')

# Categorizando as colunas
dados_precoce.set_index('EstabelecimentoIdentificador', inplace=True)
dados_precoce["EstabelecimentoMunicipio"] = dados_precoce["EstabelecimentoMunicipio"].astype('category')
dados_precoce["EstabelecimentoUF"] = dados_precoce["EstabelecimentoUF"].astype('object')
dados_precoce["IdentificadorLoteSituacaoLote"] = dados_precoce["IdentificadorLoteSituacaoLote"].astype('category')
dados_precoce["EhNovilhoPrecoce"] = dados_precoce["EhNovilhoPrecoce"].astype('category')
dados_precoce["Tipificacao"] = dados_precoce["Tipificacao"].astype('category')
dados_precoce["Maturidade"] = dados_precoce["Maturidade"].astype('category')
dados_precoce["Acabamento"] = dados_precoce["Acabamento"].astype('int64')
dados_precoce["AprovacaoCarcacaSif"] = dados_precoce["AprovacaoCarcacaSif"].astype('category')



# dados_precoce.head(10)
print(dados_precoce.dtypes)
print(dados_precoce.shape)
print(dados_precoce.describe(include=['number']))
print(dados_precoce.describe(include=['object', 'category']))

ax = plt.axes()
sns.countplot(x='Acabamento', data=dados_precoce, ax=ax)
ax.set_title('Distribuição da classe alvo (Y)')
# plt.show()

f, axarr = plt.subplots(2, 2, figsize=(15, 15))

sns.boxplot(x='Maturidade', y='Acabamento', data=dados_precoce, showmeans=True, ax=axarr[0, 0])
sns.boxplot(x='Tipificacao', y='Acabamento', data=dados_precoce, showmeans=True, ax=axarr[0, 1])

axarr[0, 0].set_title('Maturidade')
axarr[0, 1].set_title('Tipificacao')
axarr[1, 0].set_title('Data do Abate')
axarr[1, 1].set_title('Cidade')

plt.tight_layout()
plt.show()

