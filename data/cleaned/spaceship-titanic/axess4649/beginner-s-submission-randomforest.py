import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train
train.info
train['Transported'] = train['Transported'] * 1
train['CryoSleep'] = train['CryoSleep'] * 1
train['VIP'] = train['VIP'] * 1
train['HomePlanet'] = train['HomePlanet'].map({'Europa': 1, 'Earth': 2}).fillna(3)
train['Destination'] = train['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2}).fillna(3)
train['CryoSleep'] = train['CryoSleep'].fillna(3)
train['Age'] = train['Age'].fillna(train['Age'].median())
train['VIP'] = train['VIP'].fillna(3)
train['RoomService'] = train['RoomService'].fillna(train['RoomService'].median())
train['FoodCourt'] = train['FoodCourt'].fillna(train['FoodCourt'].median())
train['ShoppingMall'] = train['ShoppingMall'].fillna(train['ShoppingMall'].median())
train['Spa'] = train['Spa'].fillna(train['Spa'].median())
train['VRDeck'] = train['VRDeck'].fillna(train['VRDeck'].median())
del train['Name']
train_transported = train['Transported']
del train['Transported']
train['Destination'] = train['Destination'] / train['Destination'].max()
train['RoomService'] = train['RoomService'] / train['RoomService'].max()
train['HomePlanet'] = train['HomePlanet'] / train['HomePlanet'].max()
train['FoodCourt'] = train['FoodCourt'] / train['FoodCourt'].max()
train['ShoppingMall'] = train['ShoppingMall'] / train['ShoppingMall'].max()
train['Spa'] = train['Spa'] / train['Spa'].max()
train['VRDeck'] = train['VRDeck'] / train['VRDeck'].max()
train['Deck'] = train['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
train['Num'] = train['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
train['Side'] = train['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
del train['Cabin']
train['Deck'] = train['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}).fillna(9)
train['Deck'] = train['Deck'] / train['Deck'].max()
train['Num'] = train['Num'].fillna(train['Num'].median())
train['Num'] = train['Num'] / train['Num'].max()
train['Side'] = train['Side'].map({'P': 0, 'S': 1}).fillna(0.5)
train['CryoSleep'] = train['CryoSleep'] / train['CryoSleep'].max()
train['VIP'] = train['VIP'] / train['VIP'].max()
train['Age'] = train['Age'] / train['Age'].max()
train
test['CryoSleep'] = test['CryoSleep'] * 1
test['VIP'] = test['VIP'] * 1
test['HomePlanet'] = test['HomePlanet'].map({'Europa': 1, 'Earth': 2}).fillna(3)
test['Destination'] = test['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2}).fillna(3)
test['CryoSleep'] = test['CryoSleep'].fillna(3)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['VIP'] = test['VIP'].fillna(3)
test['RoomService'] = test['RoomService'].fillna(test['RoomService'].median())
test['FoodCourt'] = test['FoodCourt'].fillna(test['FoodCourt'].median())
test['ShoppingMall'] = test['ShoppingMall'].fillna(test['ShoppingMall'].median())
test['Spa'] = test['Spa'].fillna(test['Spa'].median())
test['VRDeck'] = test['VRDeck'].fillna(test['VRDeck'].median())
del test['Name']
test['Destination'] = test['Destination'] / test['Destination'].max()
test['RoomService'] = test['RoomService'] / test['RoomService'].max()
test['HomePlanet'] = test['HomePlanet'] / test['HomePlanet'].max()
test['FoodCourt'] = test['FoodCourt'] / test['FoodCourt'].max()
test['ShoppingMall'] = test['ShoppingMall'] / test['ShoppingMall'].max()
test['Spa'] = test['Spa'] / test['Spa'].max()
test['VRDeck'] = test['VRDeck'] / test['VRDeck'].max()
test['Deck'] = test['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
test['Num'] = test['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
test['Side'] = test['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
del test['Cabin']
test['Deck'] = test['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}).fillna(9)
test['Deck'] = test['Deck'] / test['Deck'].max()
test['Num'] = test['Num'].fillna(test['Num'].median())
test['Num'] = test['Num'] / test['Num'].max()
test['Side'] = test['Side'].map({'P': 0, 'S': 1}).fillna(0.5)
test['CryoSleep'] = test['CryoSleep'] / test['CryoSleep'].max()
test['VIP'] = test['VIP'] / test['VIP'].max()
test['Age'] = test['Age'] / test['Age'].max()
test
train.max()
train.isnull().any()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(train, train_transported, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()