import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
sns.heatmap(_input1.corr(), annot=True)
_input1.isnull().sum()
train_data1 = _input1.copy(deep=True)
data_cleaner = [train_data1, _input0]
PassengerID = _input0.PassengerId
for dataset in data_cleaner:
    dataset['HomePlanet'] = dataset['HomePlanet'].fillna(dataset['HomePlanet'].mode()[0], inplace=False)
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(dataset['CryoSleep'].mode()[0], inplace=False)
    dataset['Destination'] = dataset['Destination'].fillna(dataset['Destination'].mode()[0], inplace=False)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean(), inplace=False)
    dataset['VIP'] = dataset['VIP'].fillna(dataset['VIP'].mode()[0], inplace=False)
    dataset['RoomService'] = dataset['RoomService'].fillna(dataset['RoomService'].mean(), inplace=False)
    dataset['Spa'] = dataset['Spa'].fillna(dataset['Spa'].mean(), inplace=False)
    dataset['VRDeck'] = dataset['VRDeck'].fillna(dataset['VRDeck'].mean(), inplace=False)
    dataset = dataset.drop(['FoodCourt', 'ShoppingMall', 'Name', 'PassengerId', 'Cabin'], axis=1, inplace=False)
_input0.isnull().sum()
train_data1 = pd.get_dummies(data=train_data1, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], drop_first=True)
_input0 = pd.get_dummies(data=_input0, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], drop_first=True)
Target = train_data1.Transported.astype(int)
train_data1 = train_data1.drop(['Transported'], axis=1, inplace=False)
train_data1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()