import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()
_input1.isnull().sum()
x = _input1.iloc[:, :-1]
x
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
for D1 in _input1.columns:
    if _input1[D1].dtype == object:
        _input1[D1] = _input1[D1].astype(str)
        X1 = _input1[D1].unique()
        X2 = X1[-2]
        _input1[D1] = _input1[D1].fillna(X2)
        _input1[D1] = l1.fit_transform(_input1[D1])
    elif _input1[D1].dtype == float or _input1[D1].dtype == int:
        m1 = _input1[D1].mean()
        m1 = round(m1)
        _input1[D1] = _input1[D1].fillna(m1)
print(_input1)
del _input1['Transported']
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
for D1 in _input0.columns:
    if _input0[D1].dtype == object:
        _input0[D1] = _input0[D1].astype(str)
        X1 = _input0[D1].unique()
        X2 = X1[-2]
        _input0[D1] = _input0[D1].fillna(X2)
        _input0[D1] = l1.fit_transform(_input0[D1])
    elif _input0[D1].dtype == float or _input0[D1].dtype == int:
        m1 = _input0[D1].mean()
        m1 = round(m1)
        _input0[D1] = _input0[D1].fillna(m1)
print(_input0)
_input0
_input1.info()
_input0.info()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
from sklearn.preprocessing import LabelEncoder
l2 = LabelEncoder()
_input1['Transported'] = l2.fit_transform(_input1['Transported'])
_input1['Transported']
x = _input1.iloc[:, :].values
y = _input1.iloc[:, -1].values
x
y
data1 = _input1.iloc[:, :].values
from sklearn.preprocessing import StandardScaler
s1 = StandardScaler()
x = s1.fit_transform(x)
data1 = s1.fit_transform(data1)
_input0.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=9)
from sklearn.linear_model import LogisticRegression
l3 = LogisticRegression()