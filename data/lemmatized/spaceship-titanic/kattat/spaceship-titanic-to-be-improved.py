import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
bg_color = 'white'
ktcolors = ['#d0384e', '#ee6445', '#fa9b58', '#fece7c', '#fff1a8', '#f4faad', '#d1ed9c', '#97d5a4', '#5cb7aa', '#3682ba']
sns.set(rc={'font.style': 'normal', 'axes.facecolor': bg_color, 'figure.facecolor': bg_color, 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.grid': False, 'axes.labelsize': 20, 'figure.figsize': (5.0, 5.0), 'xtick.labelsize': 10, 'font.size': 10, 'ytick.labelsize': 10})
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input1.info()
_input1.CryoSleep = _input1.CryoSleep.astype(bool)
_input1.VIP = _input1.VIP.astype(bool)
_input1.info()
_input0.CryoSleep = _input0.CryoSleep.astype(bool)
_input0.VIP = _input0.VIP.astype(bool)
_input1.sample(15)
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)
_input1.HomePlanet.unique()
_input1.HomePlanet = _input1.HomePlanet.astype('category')
_input0.HomePlanet = _input0.HomePlanet.astype('category')
_input1.info()
_input1.Destination.unique()
_input1.Destination = _input1.Destination.astype('category')
_input0.Destination = _input0.Destination.astype('category')
_input1.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input1.isnull().sum()
pricing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in pricing_cols:
    _input1[col] = _input1[col].fillna(0, inplace=False)
    _input0[col] = _input0[col].fillna(0, inplace=False)
_input1.isnull().sum()
for col in _input1.isnull().sum().index[0:-1]:
    temp = _input1[col].value_counts().index[0]
    _input1[col] = _input1[col].fillna(temp)
    _input0[col] = _input0[col].fillna(temp)
print(f'Training NaNs:\n{_input1.isnull().sum()}\n\nTesting NaNs:\n{_input0.isnull().sum()}')
print(f'\nThe data contains {_input1.isnull().sum().sum() + _input0.isnull().sum().sum()} NaNs')
_input1 = pd.get_dummies(data=_input1, columns=['HomePlanet', 'Destination'])
_input1.columns
_input0 = pd.get_dummies(data=_input0, columns=['HomePlanet', 'Destination'])
_input0.columns
features = ['HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'CryoSleep', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
y = _input1.Transported
X = _input1[features]
X_test = _input0[features]
model = RandomForestClassifier()