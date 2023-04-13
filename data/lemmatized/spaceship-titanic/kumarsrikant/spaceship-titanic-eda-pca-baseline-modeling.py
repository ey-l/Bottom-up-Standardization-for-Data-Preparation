import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input0
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
len(_input1)
_input1.isnull().sum()
len(_input1.dropna())
6606 / 8693 * 100
_input0.isnull().sum()
_input1.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x='Transported', y='Age', data=_input1)
_input1['Age'] = _input1['Age'].fillna(int(_input1['Age'].mode()), inplace=False)
sns.boxplot(x='Transported', y='Age', data=_input1)
_input0['Age'] = _input0['Age'].fillna(int(_input0['Age'].mode()), inplace=False)
_input1[['RoomService']].isnull().sum()
sns.boxplot(x='Transported', y='RoomService', data=_input1)
_input1['RoomService'].describe()
_input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
_input0['RoomService'].describe()
_input0['RoomService'] = _input0['RoomService'].fillna(0, inplace=False)
_input1['FoodCourt'].describe()
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
_input1['ShoppingMall'].describe()
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
_input1['Spa'].describe()
_input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
_input1['VRDeck'].describe()
_input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
_input0['ShoppingMall'].describe()
_input0['FoodCourt'].describe()
_input0['Spa'].describe()
_input0['VRDeck'].describe()
_input0['VRDeck'] = _input0['VRDeck'].fillna(0, inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(0, inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0, inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0, inplace=False)
_input1.isnull().sum()
_input1['Name']
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input1.isnull().sum()
_input0['VIP'].describe()
_input1['VIP'].describe()
_input1['VIP'].value_counts().index[0]
most_freq_train = _input1['VIP'].value_counts().index[0]
most_freq_test = _input0['VIP'].value_counts().index[0]
_input1['VIP'] = _input1['VIP'].fillna(most_freq_train, inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(most_freq_test, inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('None', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('A/-1/A', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('None', inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('None', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False, inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('A/-1/A', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna('None', inplace=False)
_input1.isnull().sum()
_input1['Transported'].value_counts()
_input1
_input1.info()
from sklearn.preprocessing import MinMaxScaler