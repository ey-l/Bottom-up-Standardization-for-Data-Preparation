import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.shape
_input0.shape
_input0.dtypes
_input1.info()
_input1.head()
_input1.apply(lambda x: sum(x.isnull()))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
var_mod = ['HomePlanet', 'CryoSleep', 'VIP', 'Transported']
le = LabelEncoder()
for i in var_mod:
    _input1[i] = le.fit_transform(_input1[i])
_input1.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
var_mod = ['HomePlanet', 'CryoSleep', 'VIP']
le = LabelEncoder()
for i in var_mod:
    _input0[i] = le.fit_transform(_input0[i])
_input1 = _input1.fillna(0)
_input0 = _input0.fillna(0)
_input1.apply(lambda x: sum(x.isnull()))
_input0.apply(lambda x: sum(x.isnull()))
data = pd.concat([_input1, _input0], ignore_index=True)
data.apply(lambda x: sum(x.isnull()))
data.shape
_input1.head()
train1 = _input1.drop(['PassengerId', 'Cabin', 'Destination', 'Name', 'Transported'], axis=1)
test1 = _input1.Transported
ytest = _input0.drop(['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()