import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0.describe()
_input0.describe(include=object)
_input0.isna().sum()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.describe()
_input1.describe(include=object)
_input1.isna().sum()
vip = _input1.loc[_input1.VIP == True]['Transported']
rate_vip = sum(vip) / len(vip)
print(f'VIP who survived: {rate_vip}')
no_vip = _input1.loc[_input1.VIP == False]['Transported']
rate_no_vip = sum(no_vip) / len(no_vip)
print(f'VIP who survived: {rate_no_vip}')
from sklearn.ensemble import RandomForestClassifier
y = _input1['Transported']
features = ['VIP', 'CryoSleep', 'HomePlanet']
X = pd.get_dummies(_input1[features])
X_test = pd.get_dummies(_input0[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)