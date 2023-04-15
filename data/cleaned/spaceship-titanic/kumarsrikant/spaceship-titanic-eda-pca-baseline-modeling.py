import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
pd.read_csv('data/input/spaceship-titanic/test.csv')
train
test
train.drop('PassengerId', axis=1, inplace=True)
len(train)
train.isnull().sum()
len(train.dropna())
6606 / 8693 * 100
test.isnull().sum()
train.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x='Transported', y='Age', data=train)
train['Age'].fillna(int(train['Age'].mode()), inplace=True)
sns.boxplot(x='Transported', y='Age', data=train)
test['Age'].fillna(int(test['Age'].mode()), inplace=True)
train[['RoomService']].isnull().sum()
sns.boxplot(x='Transported', y='RoomService', data=train)
train['RoomService'].describe()
train['RoomService'].fillna(0, inplace=True)
test['RoomService'].describe()
test['RoomService'].fillna(0, inplace=True)
train['FoodCourt'].describe()
train['FoodCourt'].fillna(0, inplace=True)
train['ShoppingMall'].describe()
train['ShoppingMall'].fillna(0, inplace=True)
train['Spa'].describe()
train['Spa'].fillna(0, inplace=True)
train['VRDeck'].describe()
train['VRDeck'].fillna(0, inplace=True)
test['ShoppingMall'].describe()
test['FoodCourt'].describe()
test['Spa'].describe()
test['VRDeck'].describe()
test['VRDeck'].fillna(0, inplace=True)
test['Spa'].fillna(0, inplace=True)
test['FoodCourt'].fillna(0, inplace=True)
test['ShoppingMall'].fillna(0, inplace=True)
train.isnull().sum()
train['Name']
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.isnull().sum()
test['VIP'].describe()
train['VIP'].describe()
train['VIP'].value_counts().index[0]
most_freq_train = train['VIP'].value_counts().index[0]
most_freq_test = test['VIP'].value_counts().index[0]
train['VIP'].fillna(most_freq_train, inplace=True)
test['VIP'].fillna(most_freq_test, inplace=True)
train['HomePlanet'].fillna('None', inplace=True)
train['CryoSleep'].fillna(False, inplace=True)
train['Cabin'].fillna('A/-1/A', inplace=True)
train['Destination'].fillna('None', inplace=True)
test['HomePlanet'].fillna('None', inplace=True)
test['CryoSleep'].fillna(False, inplace=True)
test['Cabin'].fillna('A/-1/A', inplace=True)
test['Destination'].fillna('None', inplace=True)
train.isnull().sum()
train['Transported'].value_counts()
train
train.info()
from sklearn.preprocessing import MinMaxScaler