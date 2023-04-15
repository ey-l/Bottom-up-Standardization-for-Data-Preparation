import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
gc.enable()
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.info()
from sklearn.preprocessing import StandardScaler
s_Age = StandardScaler()
s_Room = StandardScaler()
s_Food = StandardScaler()
s_Shop = StandardScaler()
s_Spa = StandardScaler()
s_VR = StandardScaler()
train['Age'] = train['Age'].fillna(train.groupby('HomePlanet')['Age'].transform('median'))
train['s_Age'] = s_Age.fit_transform(np.array(train['Age']).reshape(-1, 1))
train['s_Room'] = s_Room.fit_transform(np.array(train['RoomService']).reshape(-1, 1))
train['s_Food'] = s_Food.fit_transform(np.array(train['FoodCourt']).reshape(-1, 1))
train['s_Shop'] = s_Shop.fit_transform(np.array(train['ShoppingMall']).reshape(-1, 1))
train['s_Spa'] = s_Spa.fit_transform(np.array(train['Spa']).reshape(-1, 1))
train['s_VR'] = s_VR.fit_transform(np.array(train['VRDeck']).reshape(-1, 1))
train[['Passenger_GRP', 'Company']] = train['PassengerId'].str.split('_', expand=True)
train['Passenger_GRP'] = train['Passenger_GRP'].astype(int)
train['Company'] = train['Company'].astype(int)
train['Has_Company'] = train['Company'].map(lambda x: 1 if x > 1 else 0)
train[['deck', 'num', 'side']] = train['Cabin'].str.split('/', expand=True)
train[['First Name', 'Last Name']] = train.Name.str.split(' ', expand=True)
train['TotalSpend'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
train['PCTRoomService'] = train['RoomService'] / train['TotalSpend']
train['PCTFoodCourt'] = train['FoodCourt'] / train['TotalSpend']
train['PCTShoppingMall'] = train['ShoppingMall'] / train['TotalSpend']
train['PCTSpa'] = train['Spa'] / train['TotalSpend']
train['PCTVRDeck'] = train['VRDeck'] / train['TotalSpend']
train[['VIP', 'CryoSleep']] = train[['VIP', 'CryoSleep']].astype(bool)
train['CryoSleep'] = train['CryoSleep'].fillna(True)
train['VIP'] = train['VIP'].fillna(True)
train['ENC_RoomService'] = pd.qcut(train['RoomService'], 15, labels=False, duplicates='drop')
train['ENC_FoodCourt'] = pd.qcut(train['FoodCourt'], 30, labels=False, duplicates='drop')
train['ENC_ShoppingMall'] = pd.qcut(train['ShoppingMall'], 24, labels=False, duplicates='drop')
train['ENC_Spa'] = pd.qcut(train['Spa'], 24, labels=False, duplicates='drop')
train['ENC_VRDeck'] = pd.qcut(train['VRDeck'], 24, labels=False, duplicates='drop')
train['ENC_TotalSpend'] = pd.qcut(train['TotalSpend'], 35, labels=False, duplicates='drop')
train['ENC_Age'] = pd.qcut(train['Age'], 20, labels=False, duplicates='drop')
train['num'] = train['num'].fillna(-1)
train['num'] = train['num'].astype('int')
train['ENC_num'] = pd.qcut(train['num'], 20, labels=False, duplicates='drop')
train['deck'] = train.deck.fillna(-1)
deck_no = {'F': 1, 'G': 2, 'E': 3, 'B': 4, 'C': 5, 'D': 6, 'A': 7, 'T': 8, -1: -1}
train['deck'] = train.deck.apply(lambda x: deck_no[x])
from sklearn.preprocessing import OneHotEncoder
enc_HP = OneHotEncoder(handle_unknown='ignore', sparse=False)
HomePlanet = enc_HP.fit_transform(train.HomePlanet.values.reshape(-1, 1))
HomePlanet = pd.DataFrame(HomePlanet, columns=['HP0', 'HP1', 'HP2', 'HP3'])
HomePlanet = HomePlanet.astype(int)
enc_D = OneHotEncoder(handle_unknown='ignore', sparse=False)
Destination = enc_D.fit_transform(train.Destination.values.reshape(-1, 1))
Destination = pd.DataFrame(Destination, columns=['D0', 'D1', 'D2', 'D3'])
Destination = Destination.astype(int)
train = pd.concat([train, HomePlanet], axis=1)
train = pd.concat([train, Destination], axis=1)
from sklearn.preprocessing import LabelEncoder
LE_side = LabelEncoder()
LE_Last = LabelEncoder()
LE_home = LabelEncoder()
LE_dest = LabelEncoder()