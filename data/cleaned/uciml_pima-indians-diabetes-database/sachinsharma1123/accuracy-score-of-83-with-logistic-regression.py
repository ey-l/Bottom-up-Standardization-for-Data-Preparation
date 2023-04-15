import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.isnull().sum()
df.corr()['Outcome']
y = df['Outcome']
x = df.drop(['Outcome'], axis=1)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=0, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
list_1 = []
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)