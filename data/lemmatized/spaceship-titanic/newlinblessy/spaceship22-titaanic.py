import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.shape
_input0.shape
_input1.info()
_input0.info()
a1 = _input1.drop(['Cabin', 'Name'], axis=1)
a1
b1 = _input0.drop(['Cabin', 'Name'], axis=1)
b1
a1.describe()
b1.describe()
a1.isna().sum()
b1.isna().sum()
a1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = a1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
b1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = b1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
a1
a1['VIP'] = a1['VIP'].fillna(False)
b1['VIP'] = b1['VIP'].fillna(False)
a1['HomePlanet'] = a1['HomePlanet'].fillna('Earth')
b1['HomePlanet'] = b1['HomePlanet'].fillna('Earth')
a1['Age'] = a1['Age'].fillna(a1['Age'].median())
b1['Age'] = b1['Age'].fillna(a1['Age'].median())
a1 = a1.set_index('PassengerId', inplace=False)
b1 = b1.set_index('PassengerId', inplace=False)
from sklearn.preprocessing import LabelEncoder
att = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'VRDeck']
for i in att:
    le = LabelEncoder()
    arr = np.concatenate((a1[i], b1[i])).astype(str)