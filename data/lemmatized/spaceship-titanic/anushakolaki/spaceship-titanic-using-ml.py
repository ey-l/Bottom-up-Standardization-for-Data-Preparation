import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
x_data = _input1.iloc[:, :-1]
x_data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for col in _input1.columns:
    if _input1[col].dtype == object:
        _input1[col] = _input1[col].astype(str)
        d1 = _input1[col].unique()
        d2 = d1[-2]
        _input1[col] = _input1[col].fillna(d2)
        _input1[col] = lb.fit_transform(_input1[col])
    elif _input1[col].dtype == float or _input1[col].dtype == int:
        m1 = _input1[col].mean()
        m1 = round(m1)
        _input1[col] = _input1[col].fillna(m1)
print(_input1)
del _input1['Transported']
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
for col in _input0.columns:
    if _input0[col].dtype == object:
        _input0[col] = _input0[col].astype(str)
        d1 = _input0[col].unique()
        d2 = d1[-2]
        _input0[col] = _input0[col].fillna(d2)
        _input0[col] = lb.fit_transform(_input0[col])
    elif _input0[col].dtype == float or _input0[col].dtype == int:
        m1 = _input0[col].mean()
        m1 = round(m1)
        _input0[col] = _input0[col].fillna(m1)
print(_input0)
_input0
_input1.info()
_input0.info()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
from sklearn.preprocessing import LabelEncoder
lb1 = LabelEncoder()
_input1['Transported'] = lb1.fit_transform(_input1['Transported'])
_input1['Transported']
x = _input1.iloc[:, :].values
y = _input1.iloc[:, -1].values
y
test_std = _input0.iloc[:, :].values
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)
test_std = std.fit_transform(test_std)
test_std.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()