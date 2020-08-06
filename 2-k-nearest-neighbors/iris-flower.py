#!/usr/bin/env python3

from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def myWeight(distance):
    sigma2 = .4
    return np.exp(-distance**2/sigma2)

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Labels:', np.unique(iris_y))

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])

# 1NN
model = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\nAccuracy of 1NN: %.2f%%' %(100 * accuracy_score(y_test, y_pred)))

# 7NN
model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7NN: %.2f%%' %(100 * accuracy_score(y_test, y_pred)))

# 7NN + weights
model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7NN + weights: %.2f%%' %(100 * accuracy_score(y_test, y_pred)))

# 7NN + custom weights
model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = myWeight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7NN + custom weights: %.2f%%' %(100 * accuracy_score(y_test, y_pred)))