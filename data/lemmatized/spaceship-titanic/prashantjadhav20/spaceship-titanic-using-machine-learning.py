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
_input1.shape
_input1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
for col in _input1.columns:
    if _input1[col].dtype == object:
        _input1[col] = _input1[col].astype(str)
        d1 = _input1[col].unique()
        d2 = d1[-2]
        _input1[col] = _input1[col].fillna(d2)
        _input1[col] = l1.fit_transform(_input1[col])
    elif _input1[col].dtype == float or _input1[col].dtype == int:
        m1 = _input1[col].mean()
        m1 = round(m1)
        _input1[col] = _input1[col].fillna(m1)
print(_input1)
_input1.info()
_input1['Transported'] = l1.fit_transform(_input1['Transported'])
_input1['Transported']
x = _input1.iloc[:, :-1].values
y = _input1.iloc[:, -1].values
y
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_scale = std.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_scale, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()