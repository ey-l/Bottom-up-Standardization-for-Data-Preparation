import re
import numpy as np
import pandas as pd
from pandas import Categorical
from xgboost import XGBRegressor, XGBClassifier
import sklearn
pd.options.display.max_rows = 6
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')

def enhance(df):
    for col in ['HomePlanet', 'Cabin', 'Destination', 'Name']:
        df[col] = df[col].astype('category')
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(_input1[col].mean())
        df[col] = df[col] / _input1[col].max()
    return df
_input1 = enhance(_input1)
_input0 = enhance(_input0)
columns = _input0.columns
X = _input1[columns]
Y = _input1['Transported']
(X_train, X_valid, Y_train, Y_valid) = sklearn.model_selection.train_test_split(X, Y, test_size=0.01, random_state=42)
X_test = _input0[columns]
xgb = XGBClassifier(n_jobs=-1, verbosity=0, random_state=42, tree_method='gpu_hist', enable_categorical=True)