import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import math
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.shape
_input1.head(20)
_input1.info()
print(_input1['PassengerId'].nunique())
print(_input1['Cabin'].nunique())
print(_input1['HomePlanet'].unique())
print(_input1['Destination'].unique())
_input1[_input1['HomePlanet'].isna()]
_input1[_input1['HomePlanet'].isna() & _input1['VIP'] == True]
sns.boxplot(data=_input1, x='VIP', y='Age', palette='mako')
_input1['Age'].mode()
_input1[_input1['Age'] == 24].shape
(fig, ax) = plt.subplots(2, 2, figsize=(20, 10))
sns.countplot(x='CryoSleep', data=_input1, ax=ax[0][0], palette='mako')
ax[0][0].set_title('Count of Passengers Being Put to Cryosleep and the ones that are not')
sns.boxplot(x='CryoSleep', y='Age', hue='Transported', data=_input1, ax=ax[0][1], palette='mako')
ax[0][1].set_title('Passengers being put to cryosleep with comparison to being transported and their age')
sns.countplot(x='VIP', data=_input1, ax=ax[1][0], palette='rocket')
ax[1][0].set_title('Count of VIP Passengers and the ones that are not')
sns.boxplot(x='VIP', y='Age', hue='Transported', data=_input1, ax=ax[1][1], palette='rocket')
ax[1][1].set_title('Passengers being put to cryosleep with comparison to being transported and their age')
_input1[_input1['Destination'].isna()]
_input1[_input1['Destination'].isna() & _input1['HomePlanet'].isna()]
sns.lmplot(x='RoomService', y='Age', hue='VIP', data=_input1, palette='mako')
sns.lmplot(x='RoomService', y='Age', hue='CryoSleep', data=_input1, palette='flare')
sns.lmplot(x='RoomService', y='Age', hue='Transported', data=_input1, palette='rocket')
sns.lmplot(x='FoodCourt', y='Age', hue='CryoSleep', data=_input1, palette='flare')
sns.lmplot(x='FoodCourt', y='Age', hue='VIP', data=_input1, palette='mako')
sns.lmplot(x='FoodCourt', y='Age', hue='Transported', data=_input1, palette='rocket')
sns.lmplot(x='ShoppingMall', y='Age', hue='CryoSleep', data=_input1, palette='flare')
sns.lmplot(x='ShoppingMall', y='Age', hue='VIP', data=_input1, palette='mako')
sns.lmplot(x='ShoppingMall', y='Age', hue='Transported', data=_input1, palette='rocket')
sns.lmplot(x='Spa', y='Age', hue='CryoSleep', data=_input1, palette='flare')
sns.lmplot(x='Spa', y='Age', hue='VIP', data=_input1, palette='mako')
sns.lmplot(x='Spa', y='Age', hue='Transported', data=_input1, palette='rocket')
sns.lmplot(x='VRDeck', y='Age', hue='CryoSleep', data=_input1, palette='flare')
sns.lmplot(x='VRDeck', y='Age', hue='VIP', data=_input1, palette='mako')
sns.lmplot(x='VRDeck', y='Age', hue='Transported', data=_input1, palette='rocket')
(fig, ax) = plt.subplots(3, figsize=(5, 20))
sns.countplot(data=_input1, x='HomePlanet', hue='Transported', ax=ax[0])
ax[0].set_title('Home Planet Distribution')
sns.countplot(data=_input1, x='Destination', hue='Transported', ax=ax[1], palette='mako')
ax[1].set_title('Destination Distribution')
sns.countplot(data=_input1, x='VIP', hue='Transported', ax=ax[2], palette='rocket')
ax[2].set_title('VIP Distribution')
_input1.info()
_input1 = _input1.drop(_input1[_input1['HomePlanet'].isna()].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['Destination'].isna()].index, inplace=False)
_input1 = _input1.drop('Cabin', inplace=False, axis=1)
_input1 = _input1.drop('Name', inplace=False, axis=1)
_input1 = _input1.drop(_input1[_input1['RoomService'] > 9500].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['FoodCourt'] > 25000].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['ShoppingMall'] > 20000].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['Spa'] > 17000].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['VRDeck'] > 19500].index, inplace=False)
_input1 = _input1.drop('PassengerId', inplace=False, axis=1)
_input1.info()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
newdf = _input1
newdf.info()
newdf = pd.get_dummies(newdf)
newdf
plt.subplots(figsize=(20, 20))
sns.heatmap(data=newdf.corr(), annot=True, cmap='Blues')
y = newdf.Transported
X = newdf[['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e']]
X
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=42, test_size=0.2, train_size=0.8)
lor = LogisticRegression(solver='liblinear', random_state=42)