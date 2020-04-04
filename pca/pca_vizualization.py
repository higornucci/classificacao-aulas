import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dados_completo = pd.read_csv('../input/trainingM.csv', encoding='utf-8', delimiter=',')
dados_completo = dados_completo.sample(frac=1).reset_index(drop=True)

Y = dados_completo.pop('classe')
X = dados_completo

pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(X)

principal_breast_Df = pd.DataFrame(data=principalComponents_breast,
                                   columns=['principal component 1', 'principal component 2'])

principal_breast_Df.tail()

print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


def imprimir_dataset(nome_col_x, nome_col_y, nome_imagem, col_x='principal component 1', col_y='principal component 2',
                     dataset=principal_breast_Df, targets=['dirtiness', 'not_viable', 'viable', 'white_bgd'],
                     colors=['r', 'g', 'b', 'y']):
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(nome_col_x, fontsize=16)
    plt.ylabel(nome_col_y, fontsize=16)
    for target, color in zip(targets, colors):
        indicesToKeep = Y == target
        plt.scatter(dataset.loc[indicesToKeep, col_x],
                    dataset.loc[indicesToKeep, col_y], c=color, s=50)

    plt.legend(targets, prop={'size': 15})
    plt.savefig(nome_imagem + '.png')


imprimir_dataset('Principal Component - 1', 'Principal Component - 2', 'pca')
imprimir_dataset('cor_gmedia', 'hog_0', 'dataset', dataset=dados_completo, col_x='cor_gmedia', col_y='hog_0')
