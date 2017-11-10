import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

cancer_data = np.genfromtxt(fname='breast-cancer-wisconsin.data', delimiter=',', dtype=float)
cancer_data = np.delete(arr=cancer_data, obj=0, axis=1)

X = cancer_data[:, range(0, 9)]
Y = cancer_data[:, 9]

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
X = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
y_train = y_train.ravel()
y_test = y_test.ravel()

for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)
for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)
