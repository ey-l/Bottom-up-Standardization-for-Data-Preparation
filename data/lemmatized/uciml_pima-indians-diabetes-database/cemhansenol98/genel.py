import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.columns
data.shape
data.info()
data.Age.median()
import matplotlib.pyplot as plt
pass
data.head()
y = data.Outcome
X = data.drop('Outcome', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
train_scale = scale.fit_transform(X_train)
test_scale = scale.fit_transform(X_test)
from sklearn.svm import SVC
model = SVC()