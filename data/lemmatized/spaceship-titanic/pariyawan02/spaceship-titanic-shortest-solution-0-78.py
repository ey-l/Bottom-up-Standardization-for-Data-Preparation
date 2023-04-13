import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2.head()
_input0.head(5)
_input1.head(2)
for i in _input1.columns:
    print({i: _input1[i].unique()})
_input1 = _input1.replace({False: 0, True: 1, 'Europa': 0, 'Earth': 1, 'Mars': 2})
_input0 = _input0.replace({False: 0, True: 1, 'Europa': 0, 'Earth': 1, 'Mars': 2})
_input1.head()
_input1.columns
_input1 = _input1.drop(columns=['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1, inplace=False)
t = _input0['PassengerId']
_input0 = _input0.drop(columns=['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1, inplace=False)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
_input1.head()
_input0.head()
_input1 = _input1.fillna(3)
y = _input1['Transported']
x = _input1.drop(columns=['Transported'], axis=1)
x.head()
_input0.head()
y = _input1['Transported']