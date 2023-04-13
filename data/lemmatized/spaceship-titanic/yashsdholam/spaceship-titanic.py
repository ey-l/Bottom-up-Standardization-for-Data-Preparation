import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input1.info()
_input0.info()
_input1.nunique()
_input0.nunique()
_input1.duplicated().sum()
_input0.duplicated().sum()
_input1.isnull().sum()
_input0.isnull().sum()
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('F/1/S', inplace=False)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(False, inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False, inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('F/1/S', inplace=False)
_input1[['Deck', 'Num', 'Side']] = _input1.Cabin.str.split('/', expand=True)
_input0[['Deck', 'Num', 'Side']] = _input0.Cabin.str.split('/', expand=True)
_input1['total_spent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['total_spent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1[['CryoSleep', 'VIP', 'Transported']] = _input1[['CryoSleep', 'VIP', 'Transported']].replace(False, 0)
_input1[['CryoSleep', 'VIP', 'Transported']] = _input1[['CryoSleep', 'VIP', 'Transported']].replace(True, 1)
_input0[['CryoSleep', 'VIP']] = _input0[['CryoSleep', 'VIP']].replace(False, 0)
_input0[['CryoSleep', 'VIP']] = _input0[['CryoSleep', 'VIP']].replace(True, 1)
_input1 = _input1.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=False)
_input0 = _input0.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=False)
fig = px.histogram(_input1, x='HomePlanet', width=800, height=500)
fig.show()
fig = px.histogram(_input1, x='Destination', width=800, height=500)
fig.show()
fig = px.histogram(_input1, x='Age')
fig.show()
fig = px.pie(_input1, values='Transported', names='HomePlanet', width=800, height=500)
fig.show()
plt.figure(figsize=(8, 8))
sns.histplot(data=_input1, x='CryoSleep', hue='Transported')
col = ('HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck')
fig = plt.figure(figsize=(15.0, 12.0))
for (i, c) in enumerate(col):
    ax = plt.subplot(2, 3, 1 + i)
    ax = sns.countplot(data=_input1, x=c, hue='Transported')
plt.figure(figsize=(15, 10))
sns.heatmap(_input1.corr(), annot=True, cmap='Blues')
cols = _input1.select_dtypes(['object']).columns
cols
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
_input1[cols] = oe.fit_transform(_input1[cols])
_input1.head()
cols1 = ['HomePlanet', 'Destination', 'Deck', 'Num', 'Side']
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
_input0[cols1] = oe.fit_transform(_input0[cols1])
_input0.head()
features = _input1.drop(['PassengerId', 'Transported'], axis=1)
target = _input1['Transported']
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(features, target, random_state=1, test_size=0.2, stratify=target)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.5)