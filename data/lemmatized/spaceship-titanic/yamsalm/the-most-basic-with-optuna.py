import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = _input1.fillna(0)
_input0 = _input0.fillna(0)
_input1.columns
num_columns = _input1.select_dtypes(include=np.number).columns.tolist()
num_columns
X_train_val = _input1[num_columns]
X_test = _input0[num_columns]
y_train_val = _input1['Transported'].astype(int)
(X_train, X_val, y_train, y_val) = train_test_split(X_train_val, y_train_val, test_size=0.33, random_state=42)
PassengerID = _input0.PassengerId
import optuna
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500), 'max_depth': trial.suggest_int('max_depth', 4, 16)}
    model = RandomForestClassifier(random_state=42, **params)