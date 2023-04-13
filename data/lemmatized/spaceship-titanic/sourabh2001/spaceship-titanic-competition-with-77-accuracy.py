import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.columns
_input1.head()
_input1.info()
_input1 = _input1.replace({True: 1, False: 0})
_input0 = _input0.replace({True: 1, False: 0})
_input1.info()
_input0.info()
msn.bar(_input1, color='red')

def fill_missing_vals(train, fill_missing):
    for col in fill_missing:
        _input1[col] = _input1[col].fillna(_input1[col].median(skipna=True), inplace=False)
    return _input1
fill_missing_vals(_input1, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Z', inplace=False)
_input1.info()
_input1['HomePlanet'].unique()
_input1['Transported'].isnull().sum()
from sklearn.preprocessing import LabelEncoder

def label_encode(df, col):
    _input1[col] = _input1[col].astype(str)
    _input1[col] = LabelEncoder().fit_transform(_input1[col])
    return _input1[col]
_input1['HomePlanet'] = label_encode(_input1, 'HomePlanet')
_input1['Destination'] = label_encode(_input1, 'Destination')
_input1['Cabin'] = label_encode(_input1, 'Cabin')
_input1.columns
X = _input1[['HomePlanet', 'Cabin', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
y = _input1['Transported']
X.info()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()