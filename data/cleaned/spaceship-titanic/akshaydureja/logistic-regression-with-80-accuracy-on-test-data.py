import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data
sns.heatmap(train_data.corr(), annot=True)
train_data.isnull().sum()
train_data1 = train_data.copy(deep=True)
data_cleaner = [train_data1, test_data]
PassengerID = test_data.PassengerId
for dataset in data_cleaner:
    dataset['HomePlanet'].fillna(dataset['HomePlanet'].mode()[0], inplace=True)
    dataset['CryoSleep'].fillna(dataset['CryoSleep'].mode()[0], inplace=True)
    dataset['Destination'].fillna(dataset['Destination'].mode()[0], inplace=True)
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['VIP'].fillna(dataset['VIP'].mode()[0], inplace=True)
    dataset['RoomService'].fillna(dataset['RoomService'].mean(), inplace=True)
    dataset['Spa'].fillna(dataset['Spa'].mean(), inplace=True)
    dataset['VRDeck'].fillna(dataset['VRDeck'].mean(), inplace=True)
    dataset.drop(['FoodCourt', 'ShoppingMall', 'Name', 'PassengerId', 'Cabin'], axis=1, inplace=True)
test_data.isnull().sum()
train_data1 = pd.get_dummies(data=train_data1, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], drop_first=True)
test_data = pd.get_dummies(data=test_data, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], drop_first=True)
Target = train_data1.Transported.astype(int)
train_data1.drop(['Transported'], axis=1, inplace=True)
train_data1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()