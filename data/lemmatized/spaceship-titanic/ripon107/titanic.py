import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1.Transported = le.fit_transform(_input1['Transported'])
_input1
_input1 = _input1.drop(['PassengerId'], axis=1, inplace=False)
_input1
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input1
_input1.VIP = le.fit_transform(_input1['VIP'])
_input1
_input1.Destination = le.fit_transform(_input1['Destination'])
_input1
_input1.CryoSleep = le.fit_transform(_input1['CryoSleep'])
_input1
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input1
_input1.HomePlanet = le.fit_transform(_input1['HomePlanet'])
_input1
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(skipna=True), inplace=False)
_input1.isnull().sum()
_input1 = _input1.dropna()
_input1
y = _input1.Transported
x = _input1.drop(['Transported'], axis=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
s_x = sc.fit_transform(x)
s_x
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(s_x, y, random_state=0, test_size=0.25)
x_train
x_train.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()