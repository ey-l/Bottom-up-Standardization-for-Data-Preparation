import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
dib = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dib.columns
X = dib.iloc[:, :-1].values
y = dib.iloc[:, -1].values
print(X.shape, y.shape)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.31, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()