import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1['Transported'] = _input1['Transported'] * 1
_input1['CryoSleep'] = _input1['CryoSleep'] * 1
_input1['VIP'] = _input1['VIP'] * 1
_input1['HomePlanet'] = _input1['HomePlanet'].map({'Europa': 1, 'Earth': 2}).fillna(3)
_input1['Destination'] = _input1['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2}).fillna(3)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(3)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input1['VIP'] = _input1['VIP'].fillna(3)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median())
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median())
del _input1['Name']
del _input1['Cabin']
train_transported = _input1['Transported']
del _input1['Transported']
_input1['Destination'] = _input1['Destination'] / _input1['Destination'].max()
_input1['RoomService'] = _input1['RoomService'] / _input1['RoomService'].max()
_input1['HomePlanet'] = _input1['HomePlanet'] / _input1['HomePlanet'].max()
_input1['FoodCourt'] = _input1['FoodCourt'] / _input1['FoodCourt'].max()
_input1['ShoppingMall'] = _input1['ShoppingMall'] / _input1['ShoppingMall'].max()
_input1['Spa'] = _input1['Spa'] / _input1['Spa'].max()
_input1['VRDeck'] = _input1['VRDeck'] / _input1['VRDeck'].max()
_input1['Transported'] = _input1['Transported'] * 1
_input1['CryoSleep'] = _input1['CryoSleep'] * 1
_input1['VIP'] = _input1['VIP'] * 1
_input1['HomePlanet'] = _input1['HomePlanet'].map({'Europa': 1, 'Earth': 2}).fillna(3)
_input1['Destination'] = _input1['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2}).fillna(3)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(3)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input1['VIP'] = _input1['VIP'].fillna(3)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median())
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median())
del _input1['Name']
del _input1['Cabin']
test_transported = _input1['Transported']
del _input1['Transported']
_input1['Destination'] = _input1['Destination'] / _input1['Destination'].max()
_input1['RoomService'] = _input1['RoomService'] / _input1['RoomService'].max()
_input1['HomePlanet'] = _input1['HomePlanet'] / _input1['HomePlanet'].max()
_input1['FoodCourt'] = _input1['FoodCourt'] / _input1['FoodCourt'].max()
_input1['ShoppingMall'] = _input1['ShoppingMall'] / _input1['ShoppingMall'].max()
_input1['Spa'] = _input1['Spa'] / _input1['Spa'].max()
_input1['VRDeck'] = _input1['VRDeck'] / _input1['VRDeck'].max()
(x_train, x_test, y_train, y_test) = train_test_split(_input1, train_transported, test_size=0.3, random_state=0)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=2, learning_rate=0.045, n_estimators=120)