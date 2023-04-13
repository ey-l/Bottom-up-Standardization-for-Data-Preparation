import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1['set'] = 'train'
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0['set'] = 'test'
df = pd.concat([_input1, _input0], ignore_index=True)
df['GroupId'] = df['PassengerId'].str.split('_').str[0]
df['Deck'] = df['Cabin'].str.split('/').str[0]
df['Num'] = df['Cabin'].str.split('/').str[1]
df['Side'] = df['Cabin'].str.split('/').str[2]
df['First'] = df['Name'].str.split(' ').str[0]
df['Family'] = df['Name'].str.split(' ').str[1]
df.CryoSleep = df.CryoSleep.astype(float)
df.VIP = df.VIP.astype(float)
df.GroupId = df.GroupId.astype(float)
GroupSize = df.groupby('GroupId').GroupId.count()
df = df.join(GroupSize, on='GroupId', rsuffix='_Size')
df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('Family').HomePlanet.transform('first'), inplace=False)
df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('GroupId').HomePlanet.transform('first'), inplace=False)
df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('First').HomePlanet.transform('first'), inplace=False)
df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('Destination').HomePlanet.transform(lambda x: x.mode()[0]), inplace=False)
df['Destination'] = df['Destination'].fillna(df.groupby('HomePlanet').Destination.transform(lambda x: x.mode()[0]), inplace=False)
df['TotalSpending'] = df.RoomService.fillna(0) + df.FoodCourt.fillna(0) + df.ShoppingMall.fillna(0) + df.Spa.fillna(0) + df.VRDeck.fillna(0)
df.loc[df.RoomService.isna() & df.FoodCourt.isna() & df.ShoppingMall.isna() & df.Spa.isna() & df.VRDeck.isna(), 'TotalSpending'] = np.nan
df.loc[(df.TotalSpending == 0) & df.CryoSleep.isna() & (df.Age > 12) & (df.Destination != 'TRAPPIST-1e'), 'CryoSleep'] = 1
df.loc[(df.TotalSpending > 0) & df.CryoSleep.isna(), 'CryoSleep'] = 0
df.CryoSleep = df.CryoSleep.fillna(0, inplace=False)
df[df.CryoSleep == 1] = df[df.CryoSleep == 1].fillna({'RoomService': 0, 'FoodCourt': 0, 'ShoppingMall': 0, 'Spa': 0, 'VRDeck': 0}, inplace=False)
df.VIP = df.VIP.fillna(0, inplace=False)
df['RoomService'] = df['RoomService'].fillna(df.groupby(['HomePlanet', 'Destination', 'CryoSleep']).RoomService.transform('median'), inplace=False)
df['FoodCourt'] = df['FoodCourt'].fillna(df.groupby(['HomePlanet', 'Destination', 'CryoSleep']).FoodCourt.transform('median'), inplace=False)
df['ShoppingMall'] = df['ShoppingMall'].fillna(df.groupby(['HomePlanet', 'Destination', 'CryoSleep']).ShoppingMall.transform('median'), inplace=False)
df['Spa'] = df['Spa'].fillna(df.groupby(['HomePlanet', 'Destination', 'CryoSleep']).Spa.transform('median'), inplace=False)
df['VRDeck'] = df['VRDeck'].fillna(df.groupby(['HomePlanet', 'Destination', 'CryoSleep']).VRDeck.transform('median'), inplace=False)
df['RoomService'] = df['RoomService'].fillna(df.groupby(['HomePlanet', 'CryoSleep']).RoomService.transform('median'), inplace=False)
df['FoodCourt'] = df['FoodCourt'].fillna(df.groupby(['HomePlanet', 'CryoSleep']).FoodCourt.transform('median'), inplace=False)
df['ShoppingMall'] = df['ShoppingMall'].fillna(df.groupby(['HomePlanet', 'CryoSleep']).ShoppingMall.transform('median'), inplace=False)
df['Spa'] = df['Spa'].fillna(df.groupby(['HomePlanet', 'CryoSleep']).Spa.transform('median'), inplace=False)
df['VRDeck'] = df['VRDeck'].fillna(df.groupby(['HomePlanet', 'CryoSleep']).VRDeck.transform('median'), inplace=False)
df['TotalSpending'] = df.RoomService.fillna(0) + df.FoodCourt.fillna(0) + df.ShoppingMall.fillna(0) + df.Spa.fillna(0) + df.VRDeck.fillna(0)
df['Indoor_Spending'] = df.Spa + df.RoomService + df.VRDeck
df['Outdoor_Spending'] = df.ShoppingMall + df.FoodCourt
df['Side'] = df['Side'].fillna(df.groupby('GroupId').Side.transform('first'), inplace=False)
df['Deck'] = df['Deck'].fillna(df.groupby('GroupId').Deck.transform(lambda x: next(iter(x.mode()), np.nan)), inplace=False)
df['Deck'] = df['Deck'].fillna(df.groupby(['HomePlanet', 'Destination']).Deck.transform(lambda x: x.mode()[0]), inplace=False)
df['Side'] = df['Side'].fillna(df.groupby(['HomePlanet', 'Destination']).Side.transform(lambda x: x.mode()[0]), inplace=False)
df['Deck'] = df['Deck'].fillna(df.groupby('HomePlanet').Deck.transform(lambda x: x.mode()[0]), inplace=False)
df['Side'] = df['Side'].fillna(df.groupby('HomePlanet').Side.transform(lambda x: x.mode()[0]), inplace=False)
df['Side'] = df.Side == 'P'
xg_age = df[['HomePlanet', 'CryoSleep', 'Age', 'VIP', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'First']].copy()
xg_age[['HomePlanet', 'CryoSleep', 'RoomService', 'VIP', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = xg_age[['HomePlanet', 'CryoSleep', 'RoomService', 'VIP', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0, inplace=False)
xg_age['First_Median'] = xg_age.groupby('First').Age.transform('mean')
xg_age.First_Median = xg_age.First_Median.fillna(xg_age.groupby('HomePlanet').First_Median.transform('mean'), inplace=False)
xg_age = xg_age.drop(columns=['First'], inplace=False)
xg_age = pd.get_dummies(xg_age)
train = xg_age[~xg_age.Age.isna()]
test = xg_age[xg_age.Age.isna()]
y_train = train.Age
X_train = train.drop(columns=['Age'])
X_test = test.drop(columns=['Age'])
from xgboost import XGBRegressor
xgreg = XGBRegressor(n_estimators=50, learning_rate=0.1)