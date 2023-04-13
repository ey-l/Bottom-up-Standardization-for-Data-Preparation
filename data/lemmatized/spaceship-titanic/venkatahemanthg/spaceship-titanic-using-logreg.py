import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input1.info()
_input1.describe()
_input1.isna().sum()
_input1.HomePlanet.value_counts(dropna=False, normalize=True)
_input1['HomePlanet'] = _input1.HomePlanet.fillna(_input1.HomePlanet.mode()[0])
_input1.HomePlanet.value_counts(dropna=False, normalize=True)
_input1.head()
_input1.CryoSleep.value_counts(dropna=False)
_input1['CryoSleep'] = _input1.CryoSleep.fillna(_input1.CryoSleep.mode()[0])
_input1 = _input1[~_input1.Cabin.isna()]
_input1.Destination.value_counts(dropna=False)
_input1['Destination'] = _input1.Destination.fillna(_input1.Destination.mode()[0])
_input1.Age.describe()
_input1['Age'] = _input1.Age.fillna(_input1.Age.median())
_input1.VIP.value_counts(dropna=False)
_input1['VIP'] = _input1.VIP.fillna(_input1.VIP.mode()[0])
_input1 = _input1[~_input1.RoomService.isna()]
_input1 = _input1[~_input1.FoodCourt.isna()]
_input1 = _input1[~_input1.ShoppingMall.isna()]
_input1 = _input1[~_input1.Spa.isna()]
_input1 = _input1[~_input1.VRDeck.isna()]
_input1.isna().sum()
_input1 = _input1[~_input1.Name.isna()]
_input1.info()
_input1.head()
_input1['CryoSleep'] = _input1.CryoSleep.astype(int)
_input1['VIP'] = _input1.VIP.astype(int)
_input1['Transported'] = _input1.Transported.astype(int)
plt.figure(figsize=(14, 8))
plt.subplot(2, 3, 1)
sns.boxplot(_input1.RoomService)
plt.subplot(2, 3, 2)
sns.boxplot(_input1.FoodCourt)
plt.subplot(2, 3, 3)
sns.boxplot(_input1.ShoppingMall)
plt.subplot(2, 3, 4)
sns.boxplot(_input1.Spa)
plt.subplot(2, 3, 5)
sns.boxplot(_input1.VRDeck)
plt.figure(figsize=(10, 6))
sns.heatmap(_input1.corr(), annot=True)
string = _input1.Cabin.str.split('/')
_input1['Deck'] = string.map(lambda string: string[0])
_input1['Num'] = string.map(lambda string: string[1])
_input1['Side'] = string.map(lambda string: string[2])
_input1.head()
_input1 = _input1.drop(columns=['PassengerId', 'Cabin', 'Name', 'Num'], axis=1)
_input1.head()
bins = [0, 18, 40, 100]
labels = ['teen', 'adult', 'senior']
_input1['Age_group'] = pd.cut(_input1.Age, bins=bins, labels=labels)
_input1 = _input1.drop(columns=['Age'])
_input1.head()
dummy1 = pd.get_dummies(_input1[['HomePlanet', 'Destination', 'Age_group', 'Side', 'Deck']], drop_first=True)
_input1 = pd.concat([_input1, dummy1], axis=1)
_input1 = _input1.drop(columns=['HomePlanet', 'Destination', 'Age_group', 'Side', 'Deck'])
_input1.head()
plt.figure(figsize=(20, 20))
sns.heatmap(_input1.corr(), annot=True)
from sklearn.model_selection import train_test_split
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=100)
X_train.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(X_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
X_train.head()
sum(_input1['Transported'] / len(_input1['Transported'])) * 100
import statsmodels.api as sm
logm1 = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())