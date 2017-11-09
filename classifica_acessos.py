from dados import carregar_acessos

X, Y = carregar_acessos()
from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(X, Y)

print(modelo.predict([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
