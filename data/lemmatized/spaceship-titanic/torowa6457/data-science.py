import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.corr()
_input1.describe
_input1 = _input1.drop(['Name'], axis=1)
_input1 = _input1.drop(['PassengerId'], axis=1)
_input1.isnull().sum()
_input1['deck'] = _input1['Cabin'].str.split('/', expand=True)[0]
_input1['num'] = _input1['Cabin'].str.split('/', expand=True)[1]
_input1['side'] = _input1['Cabin'].str.split('/', expand=True)[2]
_input1 = _input1.drop(['Cabin'], axis=1)
_input1.mode()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(bool)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['VIP'] = _input1['VIP'].fillna('False', inplace=False)
_input1['VIP'] = _input1['VIP'].astype(bool)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
_input1['deck'] = _input1['deck'].fillna('F', inplace=False)
_input1['side'] = _input1['side'].fillna('S', inplace=False)
_input1['Age'].value_counts()
_input1['Age'] = _input1['Age'].fillna(100, inplace=False)
_input1['Age'] = _input1['Age'].astype(int)
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 9))
plt.hist(_input1['Age'], bins=350, color='blue')
_input1['num'].value_counts()
_input1['num'] = _input1['num'].fillna(2000, inplace=False)
_input1['num'] = _input1['num'].astype(int)
plt.figure(figsize=(16, 9))
plt.hist(_input1['num'], bins=350, color='blue')
_input1.loc[(0 <= _input1['num']) & (_input1['num'] < 350), 'num'] = 1
_input1.loc[(350 <= _input1['num']) & (_input1['num'] < 600), 'num'] = 2
_input1.loc[(600 <= _input1['num']) & (_input1['num'] < 1500), 'num'] = 3
_input1.loc[(1500 <= _input1['num']) & (_input1['num'] < 2000), 'num'] = 4
_input1['num'].mode()
_input1.loc[_input1['num'] == 2000, 'num'] = 1
_input1.isnull().sum()
import seaborn as sns
sns.set()
(fig, axes) = plt.subplots(3, 3, figsize=(18, 15))
sns.countplot(x=_input1['HomePlanet'], hue=_input1['Transported'], ax=axes[0, 0])
sns.countplot(x=_input1['CryoSleep'], hue=_input1['Transported'], ax=axes[0, 1])
sns.countplot(x=_input1['Destination'], hue=_input1['Transported'], ax=axes[0, 2])
sns.countplot(x=_input1['VIP'], hue=_input1['Transported'], ax=axes[1, 0])
sns.countplot(x=_input1['deck'], hue=_input1['Transported'], ax=axes[1, 1])
sns.countplot(x=_input1['num'], hue=_input1['Transported'], ax=axes[1, 2])
sns.countplot(x=_input1['side'], hue=_input1['Transported'], ax=axes[2, 0])
_input1.head()
_input1 = _input1.drop(['VIP'], axis=1)
_input1 = _input1.drop(['num'], axis=1)
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'CryoSleep', 'Destination', 'deck', 'side'], drop_first=True)
_input1
from sklearn.preprocessing import StandardScaler
train_standard = StandardScaler()
train_copied = _input1.copy()