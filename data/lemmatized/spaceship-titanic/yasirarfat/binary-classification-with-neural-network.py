import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.shape)
_input1.head()
_input1.nunique().sort_values(ascending=False)
round(_input1.isnull().sum() * 100 / len(_input1), 2).sort_values(ascending=False)
y = _input1['Transported']
_input1 = _input1.drop(['PassengerId', 'Name', 'Transported'], axis=1)
for i in _input1:
    _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
obj = list(_input1.select_dtypes(['object']).columns)
obj
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in obj:
    _input1[i] = le.fit_transform(_input1[[i]])
_input1
sns.boxplot(data=_input1)
from sklearn.preprocessing import Normalizer