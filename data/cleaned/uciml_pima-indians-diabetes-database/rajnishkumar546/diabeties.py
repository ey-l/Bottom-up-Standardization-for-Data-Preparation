import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
train_data.head()
train_data.shape
train_data.info()
train_data.isnull().sum()
X = train_data.drop('Outcome', axis=1)
X.shape
y = train_data['Outcome']
y.shape
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape
y_train.shape
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)