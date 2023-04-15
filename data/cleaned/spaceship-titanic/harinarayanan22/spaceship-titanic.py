import numpy as np
import pandas as pd
df1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
df1.head()
df1.shape
df1.info()
df1['HomePlanet'].unique()
df1['Destination'].unique()
df1['CryoSleep'].unique()
df1.duplicated().sum()
df1.drop(columns=['Name', 'RoomService', 'Spa', 'VRDeck', 'ShoppingMall', 'FoodCourt', 'VIP', 'Cabin', 'Destination'], axis=1, inplace=True)
df1.head()
df1.isnull().sum()
df1['Age'].mean()
df1['Age'].median()
df1['Age'].fillna(df1['Age'].mean(), inplace=True)
df1.isnull().sum()
homeplanet = {'Europa': 1, 'Earth': 2, 'Mars': 3}
df1['HomePlanet'].replace(homeplanet, inplace=True)
df1.head()
df1.isnull().sum()
df1['HomePlanet'].median()
df1['HomePlanet'].fillna(df1['HomePlanet'].median(), inplace=True)
df1.isnull().sum()
df1['CryoSleep'].replace(False, 0, inplace=True)
df1['CryoSleep'].replace(True, 1, inplace=True)
df1.head()
df1['CryoSleep'].mean()
df1['CryoSleep'].fillna(df1['CryoSleep'].median(), inplace=True)
df1.isnull().sum()
df1['Transported'].replace(False, 0, inplace=True)
df1['Transported'].replace(True, 1, inplace=True)
df1.head()
df1.skew()
corr = df1.corr()
corr
x_train = df1.drop('Transported', axis=1)
x_train
y_train = df1.Transported
y_train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()