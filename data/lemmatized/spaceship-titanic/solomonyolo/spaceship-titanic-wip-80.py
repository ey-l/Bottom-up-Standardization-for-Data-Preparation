import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.head())
_input1['FirstName'] = _input1['Name'].str.split(' ', expand=True)[0]
_input1['Family'] = _input1['Name'].str.split(' ', expand=True)[1]
_input0['FirstName'] = _input0['Name'].str.split(' ', expand=True)[0]
_input0['Family'] = _input0['Name'].str.split(' ', expand=True)[1]
_input1['CabinDeck'] = _input1['Cabin'].str.split('/', expand=True)[0]
_input1['CabinNum'] = _input1['Cabin'].str.split('/', expand=True)[1]
_input1['CabinSide'] = _input1['Cabin'].str.split('/', expand=True)[2]
_input0['CabinDeck'] = _input0['Cabin'].str.split('/', expand=True)[0]
_input0['CabinNum'] = _input0['Cabin'].str.split('/', expand=True)[1]
_input0['CabinSide'] = _input0['Cabin'].str.split('/', expand=True)[2]
print(_input1.head())
print(_input1.isnull().sum())
_input1 = _input1.drop(['Name', 'Cabin'], axis=1)
_input0 = _input0.drop(['Name', 'Cabin'], axis=1)
print(_input1.head())
_input1.info()
_input1['CabinNum'] = pd.to_numeric(_input1['CabinNum'])
_input0['CabinNum'] = pd.to_numeric(_input0['CabinNum'])
_input1.info()
numeric_data = [column for column in _input1.select_dtypes(['int', 'float'])]
categoric_data = [column for column in _input1.select_dtypes(exclude=['int', 'float'])]
categoric_data1 = [column for column in _input0.select_dtypes(exclude=['int', 'float'])]
print(categoric_data)
for i in numeric_data:
    _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
    _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
for i in categoric_data:
    _input1[i] = _input1[i].fillna(_input1[i].value_counts().index[0], inplace=False)
for i in categoric_data1:
    _input0[i] = _input0[i].fillna(_input0[i].value_counts().index[0], inplace=False)
print(_input1.isnull().sum())
print(_input1.corr())
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
_input1[categoric_data] = encoder.fit_transform(_input1[categoric_data])
_input0[categoric_data1] = encoder.fit_transform(_input0[categoric_data1])
print(_input1[categoric_data])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
_input1[numeric_data] = scaler.fit_transform(_input1[numeric_data])
_input0[numeric_data] = scaler.fit_transform(_input0[numeric_data])
print(_input1[numeric_data])
print(_input1.corr())
X = _input1.drop('Transported', axis=1)
Y = _input1['Transported']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.1, max_depth=5, colsample_bytree=0.8, seed=27)