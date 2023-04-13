import pandas as pd
import numpy as np
import random as rnd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
PassengerId = _input0['PassengerId']
_input1 = _input1.dropna(axis=0, subset=['Transported'], inplace=False)
y_train = _input1.Transported
full_data = pd.concat([_input1, _input0]).reset_index(drop=True)
full_data[['Deck', 'Num', 'Side']] = full_data['Cabin'].str.split('/', expand=True)
full_data = full_data.drop(columns=['Num'], inplace=False)
full_data['HomePlanet'] = full_data['HomePlanet'].fillna('Earth', inplace=False)
full_data['CryoSleep'] = full_data['CryoSleep'].fillna(False, inplace=False)
full_data['Destination'] = full_data['Destination'].fillna('TRAPPIST-1e', inplace=False)
full_data['Age'] = full_data['Age'].fillna(_input1['Age'].median(), inplace=False)
full_data['VIP'] = full_data['VIP'].fillna(False, inplace=False)
full_data['RoomService'] = full_data['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
full_data['FoodCourt'] = full_data['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
full_data['ShoppingMall'] = full_data['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
full_data['Spa'] = full_data['Spa'].fillna(_input1['Spa'].median(), inplace=False)
full_data['VRDeck'] = full_data['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
full_data['Deck'] = full_data['Deck'].fillna(chr(ord('A') + rnd.randrange(7)), inplace=False)
full_data['Side'] = full_data['Side'].fillna(rnd.choice(['P', 'S']), inplace=False)
full_data['MoneySpent'] = full_data['FoodCourt'] + full_data['ShoppingMall'] + full_data['Spa'] + full_data['RoomService']
full_data['HomePlanet'] = full_data['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
full_data['Destination'] = full_data['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
full_data['CryoSleep'] = full_data['CryoSleep'].astype(int)
full_data['VIP'] = full_data['VIP'].astype(int)
full_data['Transported'] = full_data['Transported'].apply(lambda x: 0 if x == False else 1)
full_data.loc[full_data['RoomService'] == 0, 'RoomService'] = 0
full_data.loc[(full_data['RoomService'] > 0) & (full_data['RoomService'] <= 100), 'RoomService'] = 1
full_data.loc[(full_data['RoomService'] > 100) & (full_data['RoomService'] <= 750), 'RoomService'] = 2
full_data.loc[full_data['RoomService'] > 750, 'RoomService'] = 3
full_data.loc[full_data['FoodCourt'] == 0, 'FoodCourt'] = 0
full_data.loc[(full_data['FoodCourt'] > 0) & (full_data['FoodCourt'] <= 100), 'FoodCourt'] = 1
full_data.loc[(full_data['FoodCourt'] > 100) & (full_data['FoodCourt'] <= 750), 'FoodCourt'] = 2
full_data.loc[full_data['FoodCourt'] > 750, 'FoodCourt'] = 3
full_data.loc[full_data['ShoppingMall'] == 0, 'ShoppingMall'] = 0
full_data.loc[(full_data['ShoppingMall'] > 0) & (full_data['ShoppingMall'] <= 80), 'ShoppingMall'] = 1
full_data.loc[(full_data['ShoppingMall'] > 80) & (full_data['ShoppingMall'] <= 600), 'ShoppingMall'] = 2
full_data.loc[full_data['ShoppingMall'] > 600, 'ShoppingMall'] = 3
full_data.loc[full_data['Spa'] == 0, 'Spa'] = 0
full_data.loc[(full_data['Spa'] > 0) & (full_data['Spa'] <= 60), 'Spa'] = 1
full_data.loc[(full_data['Spa'] > 60) & (full_data['Spa'] <= 600), 'Spa'] = 2
full_data.loc[full_data['Spa'] > 600, 'Spa'] = 3
full_data.loc[full_data['VRDeck'] == 0, 'VRDeck'] = 0
full_data.loc[(full_data['VRDeck'] > 0) & (full_data['VRDeck'] <= 60), 'VRDeck'] = 1
full_data.loc[(full_data['VRDeck'] > 60) & (full_data['VRDeck'] <= 600), 'VRDeck'] = 2
full_data.loc[full_data['VRDeck'] > 600, 'VRDeck'] = 3
full_data.loc[full_data['MoneySpent'] == 0, 'MoneySpent'] = 0
full_data.loc[(full_data['MoneySpent'] > 0) & (full_data['MoneySpent'] <= 777), 'MoneySpent'] = 1
full_data.loc[(full_data['MoneySpent'] > 777) & (full_data['MoneySpent'] <= 1616), 'MoneySpent'] = 2
full_data.loc[full_data['MoneySpent'] > 1616, 'MoneySpent'] = 3
full_data['Deck'] = full_data['Deck'].apply(lambda x: ord(x) - ord('A'))
full_data['Side'] = full_data['Side'].map({'P': 0, 'S': 1})
full_data = full_data.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=False)
full_data.head(10)
_input1 = full_data.iloc[:len(_input1), :]
_input0 = full_data.iloc[len(_input1):, :]
_input1 = _input1.drop(['Transported'], axis=1, inplace=False)
_input0 = _input0.drop(['Transported'], axis=1, inplace=False)
model = XGBClassifier(n_estimators=40, learning_rate=0.15)
pipeline = Pipeline(steps=[('model', model)])