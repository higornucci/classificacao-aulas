import numpy as np

# separar 90% para treino e 10% para teste
from dados import carregar_acessos

X, Y = carregar_acessos()

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = np.array(resultado).reshape(9, 1) - np.array(teste_marcacoes)
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acertos = (total_de_acertos / total_de_elementos) * 100
print(taxa_de_acertos)
print(total_de_elementos)
