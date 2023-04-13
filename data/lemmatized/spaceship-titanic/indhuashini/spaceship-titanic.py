import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.describe()
_input1.info()
_input1.isnull().sum()
_input1.shape
_input1.duplicated()
_input1.duplicated().sum()
_input1['Destination'].value_counts()
_input1['Cabin'].value_counts()
_input1 = _input1.drop(columns=['Name', 'RoomService', 'Spa', 'VRDeck', 'ShoppingMall', 'FoodCourt', 'VIP', 'Cabin', 'Destination'], axis=1, inplace=False)
_input1
sns.kdeplot(_input1['Age'], color='green', shade=True)
plt.figure()
sns.countplot(_input1.Transported)
_input1.isnull().sum()
_input1['PassengerId'].head()
_input1['Transported'].value_counts()
_input1['CryoSleep'] = _input1['CryoSleep'].replace({'False': 0, 'True': 1})
_input1['Transported'] = _input1['Transported'].replace({'False': 0, 'True': 1})
_input1.head()
homeplanet = {'Europa': 1, 'Earth': 2, 'Mars': 3}
_input1['HomePlanet'] = _input1['HomePlanet'].replace(homeplanet, inplace=False)
_input1['HomePlanet'].median()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].median(), inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['PassengerId'].head()
_input1['HomePlanet'].unique()
_input1['CryoSleep'].unique()
_input1['CryoSleep'] = _input1['CryoSleep'].replace(False, 0, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace(True, 1, inplace=False)
_input1['Transported'] = _input1['Transported'].replace(False, 0, inplace=False)
_input1['Transported'] = _input1['Transported'].replace(True, 1, inplace=False)
_input1.head()
_input1['CryoSleep'].mean()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].median(), inplace=False)
_input1.isnull().sum()
_input1.dtypes
corr = _input1.corr()
corr
x_train = _input1.drop('Transported', axis=1)
x_train
y_train = _input1.Transported
y_train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()