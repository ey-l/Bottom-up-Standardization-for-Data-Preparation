import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
combined = [_input1, _input0]
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.isna().sum()
_input1.head()
_input1.describe()
_input1.head()
_input1.info()
import matplotlib.pyplot as plt
_input1[['HomePlanet', 'Transported']].groupby('HomePlanet').mean()
_input1[['CryoSleep', 'Transported']].groupby('CryoSleep').mean()
_input1[['Destination', 'Transported']].groupby('Destination').mean()
_input1[['VIP', 'Transported']].groupby('VIP').mean()
_input1 = _input1.drop(['Name'], axis=1)
_input0 = _input0.drop(['Name'], axis=1)
_input1['RoomService'].mode()
values1 = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for v in values1:
    _input1[v] = _input1[v].fillna(value=0)
for v in values1:
    _input0[v] = _input0[v].fillna(value=0)
values2 = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for v in values2:
    _input1[v] = _input1[v].fillna(method='bfill')
for v in values2:
    _input0[v] = _input0[v].fillna(method='bfill')
_input1['Age'] = _input1['Age'].fillna(value=round(_input1['Age'].mean()))
_input0['Age'] = _input0['Age'].fillna(value=round(_input1['Age'].mean()))
_input1.isnull().sum()
_input1['Transported'] = _input1['Transported'].map({False: 0, True: 1})
_input1['CryoSleep'] = _input1['CryoSleep'].map({False: 0, True: 1})
_input0['CryoSleep'] = _input0['CryoSleep'].map({False: 0, True: 1})
_input1['VIP'] = _input1['VIP'].map({False: 0, True: 1})
_input0['VIP'] = _input0['VIP'].map({False: 0, True: 1})
_input1['HomePlanet'] = _input1['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
_input0['HomePlanet'] = _input0['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
_input1['Destination'] = _input1['Destination'].map({'55 Cancri e': 1, 'PSO J318.5-22': 2, 'TRAPPIST-1e': 3})
_input0['Destination'] = _input0['Destination'].map({'55 Cancri e': 1, 'PSO J318.5-22': 2, 'TRAPPIST-1e': 3})
_input1['Cabin_deck'] = _input1['Cabin'].str.split('/').str[0]
_input1['Cabin_side'] = _input1['Cabin'].str.split('/').str[2]
_input0['Cabin_deck'] = _input0['Cabin'].str.split('/').str[0]
_input0['Cabin_side'] = _input0['Cabin'].str.split('/').str[2]
_input1 = _input1.drop(['Cabin'], axis=1)
_input0 = _input0.drop(['Cabin'], axis=1)
values = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
_input1['Cabin_deck'] = _input1['Cabin_deck'].map(values)
_input0['Cabin_deck'] = _input0['Cabin_deck'].map(values)
values2 = {'P': 1, 'S': 2}
_input1['Cabin_side'] = _input1['Cabin_side'].map(values2)
_input0['Cabin_side'] = _input0['Cabin_side'].map(values2)
_input1['MoneySpent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['MoneySpent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1 = _input1.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
_input0 = _input0.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
_input1.head()
_input1['InGroup'] = _input1['PassengerId'].str.split('_').str[0].duplicated(keep=False).map({False: 0, True: 1})
_input0['InGroup'] = _input0['PassengerId'].str.split('_').str[0].duplicated(keep=False).map({False: 0, True: 1})
_input1.head()
train_dfx = _input1.drop(['PassengerId'], axis=1)
test_dfx = _input0.drop(['PassengerId'], axis=1)
from sklearn.model_selection import train_test_split
y = train_dfx['Transported']
X = train_dfx.drop(['Transported'], axis=1)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1)