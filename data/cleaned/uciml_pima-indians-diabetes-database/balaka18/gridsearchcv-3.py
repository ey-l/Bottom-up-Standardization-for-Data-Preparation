import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.head()
data.info()
X = data.iloc[:, :8].values
Y = data.iloc[:, -1].values
scaler = StandardScaler()
scaler.fit_transform(X)
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=20, random_state=3)
dt = DecisionTreeClassifier()