import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input1.info
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
train_transported = _input1['Transported']
del _input1['Transported']
_input1['Destination'] = _input1['Destination'] / _input1['Destination'].max()
_input1['RoomService'] = _input1['RoomService'] / _input1['RoomService'].max()
_input1['HomePlanet'] = _input1['HomePlanet'] / _input1['HomePlanet'].max()
_input1['FoodCourt'] = _input1['FoodCourt'] / _input1['FoodCourt'].max()
_input1['ShoppingMall'] = _input1['ShoppingMall'] / _input1['ShoppingMall'].max()
_input1['Spa'] = _input1['Spa'] / _input1['Spa'].max()
_input1['VRDeck'] = _input1['VRDeck'] / _input1['VRDeck'].max()
_input1['Deck'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
_input1['Num'] = _input1['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
_input1['Side'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
del _input1['Cabin']
_input1['Deck'] = _input1['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}).fillna(9)
_input1['Deck'] = _input1['Deck'] / _input1['Deck'].max()
_input1['Num'] = _input1['Num'].fillna(_input1['Num'].median())
_input1['Num'] = _input1['Num'] / _input1['Num'].max()
_input1['Side'] = _input1['Side'].map({'P': 0, 'S': 1}).fillna(0.5)
_input1['CryoSleep'] = _input1['CryoSleep'] / _input1['CryoSleep'].max()
_input1['VIP'] = _input1['VIP'] / _input1['VIP'].max()
_input1['Age'] = _input1['Age'] / _input1['Age'].max()
_input1
_input0['CryoSleep'] = _input0['CryoSleep'] * 1
_input0['VIP'] = _input0['VIP'] * 1
_input0['HomePlanet'] = _input0['HomePlanet'].map({'Europa': 1, 'Earth': 2}).fillna(3)
_input0['Destination'] = _input0['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2}).fillna(3)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(3)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median())
_input0['VIP'] = _input0['VIP'].fillna(3)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].median())
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].median())
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].median())
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].median())
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].median())
del _input0['Name']
_input0['Destination'] = _input0['Destination'] / _input0['Destination'].max()
_input0['RoomService'] = _input0['RoomService'] / _input0['RoomService'].max()
_input0['HomePlanet'] = _input0['HomePlanet'] / _input0['HomePlanet'].max()
_input0['FoodCourt'] = _input0['FoodCourt'] / _input0['FoodCourt'].max()
_input0['ShoppingMall'] = _input0['ShoppingMall'] / _input0['ShoppingMall'].max()
_input0['Spa'] = _input0['Spa'] / _input0['Spa'].max()
_input0['VRDeck'] = _input0['VRDeck'] / _input0['VRDeck'].max()
_input0['Deck'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
_input0['Num'] = _input0['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
_input0['Side'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
del _input0['Cabin']
_input0['Deck'] = _input0['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}).fillna(9)
_input0['Deck'] = _input0['Deck'] / _input0['Deck'].max()
_input0['Num'] = _input0['Num'].fillna(_input0['Num'].median())
_input0['Num'] = _input0['Num'] / _input0['Num'].max()
_input0['Side'] = _input0['Side'].map({'P': 0, 'S': 1}).fillna(0.5)
_input0['CryoSleep'] = _input0['CryoSleep'] / _input0['CryoSleep'].max()
_input0['VIP'] = _input0['VIP'] / _input0['VIP'].max()
_input0['Age'] = _input0['Age'] / _input0['Age'].max()
_input0
_input1.max()
_input1.isnull().any()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(_input1, train_transported, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()