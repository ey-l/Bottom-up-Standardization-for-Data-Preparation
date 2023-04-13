import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
y = data.Outcome
X = data.drop(['Outcome'], axis=1)
X.head()
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

def model_select(X_train, X_valid, y_train, y_valid):
    test_preds = []
    for n in range(50, 401, 50):
        model = RandomForestRegressor(n_estimators=n, random_state=0)