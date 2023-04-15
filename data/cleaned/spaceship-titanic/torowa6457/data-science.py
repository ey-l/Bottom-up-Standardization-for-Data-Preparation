import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.corr()
train.describe
train = train.drop(['Name'], axis=1)
train = train.drop(['PassengerId'], axis=1)
train.isnull().sum()
train['deck'] = train['Cabin'].str.split('/', expand=True)[0]
train['num'] = train['Cabin'].str.split('/', expand=True)[1]
train['side'] = train['Cabin'].str.split('/', expand=True)[2]
train = train.drop(['Cabin'], axis=1)
train.mode()
train['HomePlanet'].fillna('Earth', inplace=True)
train['CryoSleep'].fillna('False', inplace=True)
train['CryoSleep'] = train['CryoSleep'].astype(bool)
train['Destination'].fillna('TRAPPIST-1e', inplace=True)
train['VIP'].fillna('False', inplace=True)
train['VIP'] = train['VIP'].astype(bool)
train['RoomService'].fillna(train['RoomService'].median(), inplace=True)
train['FoodCourt'].fillna(train['FoodCourt'].median(), inplace=True)
train['ShoppingMall'].fillna(train['ShoppingMall'].median(), inplace=True)
train['Spa'].fillna(train['Spa'].median(), inplace=True)
train['VRDeck'].fillna(train['VRDeck'].median(), inplace=True)
train['deck'].fillna('F', inplace=True)
train['side'].fillna('S', inplace=True)
train['Age'].value_counts()
train['Age'].fillna(100, inplace=True)
train['Age'] = train['Age'].astype(int)
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 9))
plt.hist(train['Age'], bins=350, color='blue')
train['num'].value_counts()
train['num'].fillna(2000, inplace=True)
train['num'] = train['num'].astype(int)
plt.figure(figsize=(16, 9))
plt.hist(train['num'], bins=350, color='blue')
train.loc[(0 <= train['num']) & (train['num'] < 350), 'num'] = 1
train.loc[(350 <= train['num']) & (train['num'] < 600), 'num'] = 2
train.loc[(600 <= train['num']) & (train['num'] < 1500), 'num'] = 3
train.loc[(1500 <= train['num']) & (train['num'] < 2000), 'num'] = 4
train['num'].mode()
train.loc[train['num'] == 2000, 'num'] = 1
train.isnull().sum()
import seaborn as sns
sns.set()
(fig, axes) = plt.subplots(3, 3, figsize=(18, 15))
sns.countplot(x=train['HomePlanet'], hue=train['Transported'], ax=axes[0, 0])
sns.countplot(x=train['CryoSleep'], hue=train['Transported'], ax=axes[0, 1])
sns.countplot(x=train['Destination'], hue=train['Transported'], ax=axes[0, 2])
sns.countplot(x=train['VIP'], hue=train['Transported'], ax=axes[1, 0])
sns.countplot(x=train['deck'], hue=train['Transported'], ax=axes[1, 1])
sns.countplot(x=train['num'], hue=train['Transported'], ax=axes[1, 2])
sns.countplot(x=train['side'], hue=train['Transported'], ax=axes[2, 0])
train.head()
train = train.drop(['VIP'], axis=1)
train = train.drop(['num'], axis=1)
train = pd.get_dummies(train, columns=['HomePlanet', 'CryoSleep', 'Destination', 'deck', 'side'], drop_first=True)
train
from sklearn.preprocessing import StandardScaler
train_standard = StandardScaler()
train_copied = train.copy()