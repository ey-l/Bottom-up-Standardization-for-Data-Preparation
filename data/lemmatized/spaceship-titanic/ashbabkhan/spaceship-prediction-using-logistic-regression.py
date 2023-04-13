import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.head())
print(_input1.isna().sum())
print('-------------------------')
print(_input0.isna().sum())
_input1.dtypes
for col in _input1.columns:
    if _input1[col].dtypes == 'float64':
        _input1[col] = _input1[col].fillna(_input1[col].mean(), inplace=False)
        _input0[col] = _input0[col].fillna(_input0[col].mean(), inplace=False)
    elif _input1[col].dtypes == 'object':
        _input1[col] = _input1[col].fillna(_input1[col].value_counts()[0], inplace=False)
        _input0[col] = _input0[col].fillna(_input0[col].value_counts()[0], inplace=False)
print(_input1.isna().sum())
print('-----------------------')
print(_input0.isna().sum())
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input1['VIP'] = _input1['VIP'].astype(int)
_input1['Transported'] = _input1['Transported'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
_input0['VIP'] = _input0['VIP'].astype(int)
print(_input1.head())
print('----------------------------------------------------------------')
print(_input0.head())
_input1 = pd.get_dummies(_input1)
test_data1 = pd.get_dummies(_input0)
print(_input1.shape)
print('------------')
print(test_data1.shape)
input_features = _input1.drop('Transported', axis=1)
target = _input1['Transported']
test_data1 = test_data1.reindex(columns=input_features.columns, fill_value=0)
test_data1.shape
input_features_numpy = input_features.values
test_data_numpy = test_data1.values
min_max_scaler = MinMaxScaler()
input_features_numpy = min_max_scaler.fit_transform(input_features_numpy)
test_data_numpy = min_max_scaler.fit_transform(test_data_numpy)
print(input_features_numpy[0:5])
print('------------------------------')
print(test_data_numpy[0:5])
logreg = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=5, shuffle=True, random_state=4)
cv = cross_val_score(logreg, input_features_numpy, target, cv=kf)
print(cv)