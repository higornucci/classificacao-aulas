import warnings
import time
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

dados_completo = pd.read_csv('../input/DadosCompletoTransformadoML.csv', encoding='utf-8')
dados_completo.set_index('index', inplace=True)

conjunto_treinamento, conjunto_teste = train_test_split(dados_completo, test_size=0.2, random_state=42)
print(conjunto_treinamento.head(10))

X_treino, X_teste, Y_treino, Y_teste = conjunto_treinamento.drop('acabamento', axis=1), conjunto_teste.drop('acabamento', axis=1), conjunto_treinamento['acabamento'], conjunto_teste['acabamento']
print('X Treino:', X_treino.head(10))
print('Y Treino:', Y_treino.head(10))
print('X Teste:', X_teste.head(10))
print('Y Teste:', Y_teste.head(10))

seed = 7
num_folds = 3
processors = 1
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, random_state=seed)

# preparando alguns modelos
modelos = [('SVM', SVC(decision_function_shape='ovo')),
           ('K-NN', KNeighborsClassifier(n_neighbors=5)),
           ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)),
           ('NB', MultinomialNB())]

# Validar cada um dos modelos
for nome, modelo in modelos:
    inicio = time.time()
    cv_resultados = cross_val_score(modelo, X_treino, Y_treino, cv=kfold, scoring=scoring)
    print("{0}: ({1:.4f}) +/- ({2:.3f})".format(nome, cv_resultados.mean(), cv_resultados.std()))
    modelo.fit(X_treino, Y_treino)
    preds = modelo.predict(X_teste)
    final = time.time()
    resultado = pd.DataFrame()
    resultado["id"] = X_teste.index
    resultado["item.acabamento"] = preds
    resultado.to_csv("resultado_" + nome + ".csv", encoding='utf-8', index=False)
    print('Tempo de execução do ' + nome + ':', final - inicio)
