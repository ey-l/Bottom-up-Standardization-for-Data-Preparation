import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
gc.enable()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
from sklearn.preprocessing import StandardScaler
s_Age = StandardScaler()
s_Room = StandardScaler()
s_Food = StandardScaler()
s_Shop = StandardScaler()
s_Spa = StandardScaler()
s_VR = StandardScaler()
_input1['Age'] = _input1['Age'].fillna(_input1.groupby('HomePlanet')['Age'].transform('median'))
_input1['s_Age'] = s_Age.fit_transform(np.array(_input1['Age']).reshape(-1, 1))
_input1['s_Room'] = s_Room.fit_transform(np.array(_input1['RoomService']).reshape(-1, 1))
_input1['s_Food'] = s_Food.fit_transform(np.array(_input1['FoodCourt']).reshape(-1, 1))
_input1['s_Shop'] = s_Shop.fit_transform(np.array(_input1['ShoppingMall']).reshape(-1, 1))
_input1['s_Spa'] = s_Spa.fit_transform(np.array(_input1['Spa']).reshape(-1, 1))
_input1['s_VR'] = s_VR.fit_transform(np.array(_input1['VRDeck']).reshape(-1, 1))
_input1[['Passenger_GRP', 'Company']] = _input1['PassengerId'].str.split('_', expand=True)
_input1['Passenger_GRP'] = _input1['Passenger_GRP'].astype(int)
_input1['Company'] = _input1['Company'].astype(int)
_input1['Has_Company'] = _input1['Company'].map(lambda x: 1 if x > 1 else 0)
_input1[['deck', 'num', 'side']] = _input1['Cabin'].str.split('/', expand=True)
_input1[['First Name', 'Last Name']] = _input1.Name.str.split(' ', expand=True)
_input1['TotalSpend'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1['PCTRoomService'] = _input1['RoomService'] / _input1['TotalSpend']
_input1['PCTFoodCourt'] = _input1['FoodCourt'] / _input1['TotalSpend']
_input1['PCTShoppingMall'] = _input1['ShoppingMall'] / _input1['TotalSpend']
_input1['PCTSpa'] = _input1['Spa'] / _input1['TotalSpend']
_input1['PCTVRDeck'] = _input1['VRDeck'] / _input1['TotalSpend']
_input1[['VIP', 'CryoSleep']] = _input1[['VIP', 'CryoSleep']].astype(bool)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(True)
_input1['VIP'] = _input1['VIP'].fillna(True)
_input1['ENC_RoomService'] = pd.qcut(_input1['RoomService'], 15, labels=False, duplicates='drop')
_input1['ENC_FoodCourt'] = pd.qcut(_input1['FoodCourt'], 30, labels=False, duplicates='drop')
_input1['ENC_ShoppingMall'] = pd.qcut(_input1['ShoppingMall'], 24, labels=False, duplicates='drop')
_input1['ENC_Spa'] = pd.qcut(_input1['Spa'], 24, labels=False, duplicates='drop')
_input1['ENC_VRDeck'] = pd.qcut(_input1['VRDeck'], 24, labels=False, duplicates='drop')
_input1['ENC_TotalSpend'] = pd.qcut(_input1['TotalSpend'], 35, labels=False, duplicates='drop')
_input1['ENC_Age'] = pd.qcut(_input1['Age'], 20, labels=False, duplicates='drop')
_input1['num'] = _input1['num'].fillna(-1)
_input1['num'] = _input1['num'].astype('int')
_input1['ENC_num'] = pd.qcut(_input1['num'], 20, labels=False, duplicates='drop')
_input1['deck'] = _input1.deck.fillna(-1)
deck_no = {'F': 1, 'G': 2, 'E': 3, 'B': 4, 'C': 5, 'D': 6, 'A': 7, 'T': 8, -1: -1}
_input1['deck'] = _input1.deck.apply(lambda x: deck_no[x])
from sklearn.preprocessing import OneHotEncoder
enc_HP = OneHotEncoder(handle_unknown='ignore', sparse=False)
HomePlanet = enc_HP.fit_transform(_input1.HomePlanet.values.reshape(-1, 1))
HomePlanet = pd.DataFrame(HomePlanet, columns=['HP0', 'HP1', 'HP2', 'HP3'])
HomePlanet = HomePlanet.astype(int)
enc_D = OneHotEncoder(handle_unknown='ignore', sparse=False)
Destination = enc_D.fit_transform(_input1.Destination.values.reshape(-1, 1))
Destination = pd.DataFrame(Destination, columns=['D0', 'D1', 'D2', 'D3'])
Destination = Destination.astype(int)
_input1 = pd.concat([_input1, HomePlanet], axis=1)
_input1 = pd.concat([_input1, Destination], axis=1)
from sklearn.preprocessing import LabelEncoder
LE_side = LabelEncoder()
LE_Last = LabelEncoder()
LE_home = LabelEncoder()
LE_dest = LabelEncoder()