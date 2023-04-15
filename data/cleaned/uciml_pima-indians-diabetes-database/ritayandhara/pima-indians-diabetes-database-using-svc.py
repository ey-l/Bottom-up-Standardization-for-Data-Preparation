import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.sample(4)
df.info()
df.describe()
df.isnull().sum()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=506)
from sklearn.svm import SVC
svc = SVC()
from sklearn.model_selection import GridSearchCV
k = ['rbf', 'linear', 'poly', 'sigmoid']
c = range(1, 5)
param_grid = dict(kernel=k, C=c)
grid_svc = GridSearchCV(svc, param_grid=param_grid, cv=10, n_jobs=-1)