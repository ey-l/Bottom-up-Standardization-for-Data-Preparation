import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab
diab.info()
diab.isnull().sum()
corr = diab.corr()
pass
from sklearn.model_selection import train_test_split
x = diab.drop(['Outcome'], axis=1)
y = diab['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
pass
from sklearn.neighbors import KNeighborsClassifier