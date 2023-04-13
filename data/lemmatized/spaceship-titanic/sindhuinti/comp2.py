import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.isnull().sum()
_input1 = _input1.drop(['Cabin', 'Name'], axis=1)
_input0 = _input0.drop(['Cabin', 'Name'], axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
_input1 = pd.DataFrame(imputer.fit_transform(_input1), columns=_input1.columns)
_input0 = pd.DataFrame(imputer.fit_transform(_input0), columns=_input0.columns)
_input1['CryoSleep'] = _input1['CryoSleep'].astype('int')
_input0['CryoSleep'] = _input0['CryoSleep'].astype('int')
_input1['VIP'] = _input1['VIP'].astype('int')
_input0['VIP'] = _input0['VIP'].astype('int')
_input1['Transported'] = _input1['Transported'].astype('int')
_input1.head()
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'Destination'])
_input0 = pd.get_dummies(_input0, columns=['HomePlanet', 'Destination'])
_input1.isnull().sum()
X_train = _input1.drop(['Transported'], axis=1)
y_train = _input1['Transported']
X_test = _input0
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()