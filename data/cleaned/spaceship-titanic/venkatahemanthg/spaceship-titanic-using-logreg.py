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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
train.info()
train.describe()
train.isna().sum()
train.HomePlanet.value_counts(dropna=False, normalize=True)
train['HomePlanet'] = train.HomePlanet.fillna(train.HomePlanet.mode()[0])
train.HomePlanet.value_counts(dropna=False, normalize=True)
train.head()
train.CryoSleep.value_counts(dropna=False)
train['CryoSleep'] = train.CryoSleep.fillna(train.CryoSleep.mode()[0])
train = train[~train.Cabin.isna()]
train.Destination.value_counts(dropna=False)
train['Destination'] = train.Destination.fillna(train.Destination.mode()[0])
train.Age.describe()
train['Age'] = train.Age.fillna(train.Age.median())
train.VIP.value_counts(dropna=False)
train['VIP'] = train.VIP.fillna(train.VIP.mode()[0])
train = train[~train.RoomService.isna()]
train = train[~train.FoodCourt.isna()]
train = train[~train.ShoppingMall.isna()]
train = train[~train.Spa.isna()]
train = train[~train.VRDeck.isna()]
train.isna().sum()
train = train[~train.Name.isna()]
train.info()
train.head()
train['CryoSleep'] = train.CryoSleep.astype(int)
train['VIP'] = train.VIP.astype(int)
train['Transported'] = train.Transported.astype(int)
plt.figure(figsize=(14, 8))
plt.subplot(2, 3, 1)
sns.boxplot(train.RoomService)
plt.subplot(2, 3, 2)
sns.boxplot(train.FoodCourt)
plt.subplot(2, 3, 3)
sns.boxplot(train.ShoppingMall)
plt.subplot(2, 3, 4)
sns.boxplot(train.Spa)
plt.subplot(2, 3, 5)
sns.boxplot(train.VRDeck)
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True)
string = train.Cabin.str.split('/')
train['Deck'] = string.map(lambda string: string[0])
train['Num'] = string.map(lambda string: string[1])
train['Side'] = string.map(lambda string: string[2])
train.head()
train = train.drop(columns=['PassengerId', 'Cabin', 'Name', 'Num'], axis=1)
train.head()
bins = [0, 18, 40, 100]
labels = ['teen', 'adult', 'senior']
train['Age_group'] = pd.cut(train.Age, bins=bins, labels=labels)
train = train.drop(columns=['Age'])
train.head()
dummy1 = pd.get_dummies(train[['HomePlanet', 'Destination', 'Age_group', 'Side', 'Deck']], drop_first=True)
train = pd.concat([train, dummy1], axis=1)
train = train.drop(columns=['HomePlanet', 'Destination', 'Age_group', 'Side', 'Deck'])
train.head()
plt.figure(figsize=(20, 20))
sns.heatmap(train.corr(), annot=True)
from sklearn.model_selection import train_test_split
X = train.drop('Transported', axis=1)
y = train['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=100)
X_train.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(X_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
X_train.head()
sum(train['Transported'] / len(train['Transported'])) * 100
import statsmodels.api as sm
logm1 = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())