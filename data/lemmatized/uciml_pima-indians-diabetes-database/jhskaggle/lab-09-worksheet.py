import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import sklearn.model_selection as model_selection
from sklearn import tree
from sklearn import metrics
df2 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df2.head(10)
X = df2.iloc[:, 0:8]
y = df2.iloc[:, 8]
print(X)
(X_train, X_test, y_train, y_test) = model_selection.train_test_split(X, y, test_size=0.2, random_state=4)
(X_train_new, X_val, y_train_new, y_val) = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=4)
from sklearn.model_selection import GridSearchCV
clf_3 = KNeighborsClassifier()
param_grid = [{'weights': ['uniform'], 'n_neighbors': list(range(1, 30))}, {'weights': ['distance'], 'n_neighbors': list(range(1, 30))}]
print(param_grid)