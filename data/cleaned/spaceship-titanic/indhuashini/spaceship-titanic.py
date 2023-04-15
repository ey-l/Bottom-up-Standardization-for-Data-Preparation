import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train
train.describe()
train.info()
train.isnull().sum()
train.shape
train.duplicated()
train.duplicated().sum()
train['Destination'].value_counts()
train['Cabin'].value_counts()
train.drop(columns=['Name', 'RoomService', 'Spa', 'VRDeck', 'ShoppingMall', 'FoodCourt', 'VIP', 'Cabin', 'Destination'], axis=1, inplace=True)
train
sns.kdeplot(train['Age'], color='green', shade=True)

plt.figure()
sns.countplot(train.Transported)
train.isnull().sum()
train['PassengerId'].head()
train['Transported'].value_counts()
train['CryoSleep'] = train['CryoSleep'].replace({'False': 0, 'True': 1})
train['Transported'] = train['Transported'].replace({'False': 0, 'True': 1})
train.head()
homeplanet = {'Europa': 1, 'Earth': 2, 'Mars': 3}
train['HomePlanet'].replace(homeplanet, inplace=True)
train['HomePlanet'].median()
train['HomePlanet'].fillna(train['HomePlanet'].median(), inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['PassengerId'].head()
train['HomePlanet'].unique()
train['CryoSleep'].unique()
train['CryoSleep'].replace(False, 0, inplace=True)
train['CryoSleep'].replace(True, 1, inplace=True)
train['Transported'].replace(False, 0, inplace=True)
train['Transported'].replace(True, 1, inplace=True)
train.head()
train['CryoSleep'].mean()
train['CryoSleep'].fillna(train['CryoSleep'].median(), inplace=True)
train.isnull().sum()
train.dtypes
corr = train.corr()
corr
x_train = train.drop('Transported', axis=1)
x_train
y_train = train.Transported
y_train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()