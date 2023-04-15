import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
combined = [train_df, test_df]
sample_sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_df.isna().sum()
train_df.head()
train_df.describe()
train_df.head()
train_df.info()
import matplotlib.pyplot as plt
train_df[['HomePlanet', 'Transported']].groupby('HomePlanet').mean()
train_df[['CryoSleep', 'Transported']].groupby('CryoSleep').mean()
train_df[['Destination', 'Transported']].groupby('Destination').mean()
train_df[['VIP', 'Transported']].groupby('VIP').mean()
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df['RoomService'].mode()
values1 = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for v in values1:
    train_df[v] = train_df[v].fillna(value=0)
for v in values1:
    test_df[v] = test_df[v].fillna(value=0)
values2 = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for v in values2:
    train_df[v] = train_df[v].fillna(method='bfill')
for v in values2:
    test_df[v] = test_df[v].fillna(method='bfill')
train_df['Age'] = train_df['Age'].fillna(value=round(train_df['Age'].mean()))
test_df['Age'] = test_df['Age'].fillna(value=round(train_df['Age'].mean()))
train_df.isnull().sum()
train_df['Transported'] = train_df['Transported'].map({False: 0, True: 1})
train_df['CryoSleep'] = train_df['CryoSleep'].map({False: 0, True: 1})
test_df['CryoSleep'] = test_df['CryoSleep'].map({False: 0, True: 1})
train_df['VIP'] = train_df['VIP'].map({False: 0, True: 1})
test_df['VIP'] = test_df['VIP'].map({False: 0, True: 1})
train_df['HomePlanet'] = train_df['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
test_df['HomePlanet'] = test_df['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
train_df['Destination'] = train_df['Destination'].map({'55 Cancri e': 1, 'PSO J318.5-22': 2, 'TRAPPIST-1e': 3})
test_df['Destination'] = test_df['Destination'].map({'55 Cancri e': 1, 'PSO J318.5-22': 2, 'TRAPPIST-1e': 3})
train_df['Cabin_deck'] = train_df['Cabin'].str.split('/').str[0]
train_df['Cabin_side'] = train_df['Cabin'].str.split('/').str[2]
test_df['Cabin_deck'] = test_df['Cabin'].str.split('/').str[0]
test_df['Cabin_side'] = test_df['Cabin'].str.split('/').str[2]
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
values = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
train_df['Cabin_deck'] = train_df['Cabin_deck'].map(values)
test_df['Cabin_deck'] = test_df['Cabin_deck'].map(values)
values2 = {'P': 1, 'S': 2}
train_df['Cabin_side'] = train_df['Cabin_side'].map(values2)
test_df['Cabin_side'] = test_df['Cabin_side'].map(values2)
train_df['MoneySpent'] = train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck']
test_df['MoneySpent'] = test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck']
train_df = train_df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
test_df = test_df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
train_df.head()
train_df['InGroup'] = train_df['PassengerId'].str.split('_').str[0].duplicated(keep=False).map({False: 0, True: 1})
test_df['InGroup'] = test_df['PassengerId'].str.split('_').str[0].duplicated(keep=False).map({False: 0, True: 1})
train_df.head()
train_dfx = train_df.drop(['PassengerId'], axis=1)
test_dfx = test_df.drop(['PassengerId'], axis=1)
from sklearn.model_selection import train_test_split
y = train_dfx['Transported']
X = train_dfx.drop(['Transported'], axis=1)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1)