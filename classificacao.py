import numpy as np

# [é gordo?, perna curta?, late?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = np.array([porco1, porco2, porco3, cachorro1, cachorro2, cachorro3])
marcacoes = np.array([1, 1, 1, -1, -1, -1])
misterioso = [[1, 1, 1], [1, 0, 0], [0, 0, 1]]
marcacoes_teste = [-1, 1, -1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

resultado = modelo.predict(misterioso)
print(resultado)
diferencas = resultado - marcacoes_teste
acertos = [d for d in diferencas if d == 0]
taxa_de_acertos = len(acertos) / len(misterioso)
print(taxa_de_acertos * 100)
