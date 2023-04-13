import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
myData = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
myData.head(10)
myData.describe()
myData.isnull().sum()
X = myData.iloc[:, :-1]
y = myData.iloc[:, 8]
X
y
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)
KNN = KNeighborsClassifier(n_neighbors=3)