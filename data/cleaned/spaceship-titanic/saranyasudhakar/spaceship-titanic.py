import numpy as np
import pandas as pd
import os
ship_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
ship_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
ship_train.head()
ship_test.head()
sample_submission.head()
ship_train.shape
ship_train.describe()
ship_train.isnull().sum()
ship_train.describe()
ship_train['HomePlanet'].value_counts()
ship_train['Transported'].value_counts()
ship_train['Destination'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(ship_train['Age'])
sns.barplot(data=ship_train, x='VIP', y='Transported')
sns.barplot(data=ship_train, x='CryoSleep', y='Transported')
sns.scatterplot(data=ship_train, x='VIP', y='RoomService')
ship_train.isnull().sum()
ship_test.isnull().sum()
ship_train['HomePlanet'].mode()
ship_train['HomePlanet'] = ship_train['HomePlanet'].fillna('Earth')
ship_test['HomePlanet'] = ship_test['HomePlanet'].fillna('Earth')
ship_train['CryoSleep'].mode()
ship_train['CryoSleep'] = ship_train['CryoSleep'].fillna('False')
ship_test['CryoSleep'] = ship_test['CryoSleep'].fillna('False')
ship_train['Destination'].mode()
ship_train['Destination'] = ship_train['Destination'].fillna('TRAPPIST-1e')
ship_test['Destination'] = ship_test['Destination'].fillna('TRAPPIST-1e')
age_mean = ship_train['Age'].mean()
age_mean
ship_train['Age'] = ship_train['Age'].fillna(round(age_mean))
ship_test['Age'] = ship_test['Age'].fillna(round(age_mean))
ship_train['VIP'].mode()
ship_train['VIP'] = ship_train['VIP'].fillna('False')
ship_test['VIP'] = ship_test['VIP'].fillna('False')
ship_train['RoomService'].mode()
ship_train['RoomService'] = ship_train['RoomService'].fillna(0.0)
ship_test['RoomService'] = ship_test['RoomService'].fillna(0.0)
ship_train['FoodCourt'].mode()
ship_train['FoodCourt'] = ship_train['FoodCourt'].fillna(0.0)
ship_test['FoodCourt'] = ship_test['FoodCourt'].fillna(0.0)
ship_train['ShoppingMall'].value_counts()
ship_train['ShoppingMall'] = ship_train['ShoppingMall'].fillna(0.0)
ship_test['ShoppingMall'] = ship_test['ShoppingMall'].fillna(0.0)
ship_train['Spa'].value_counts()
ship_train['Spa'] = ship_train['Spa'].fillna(0.0)
ship_test['Spa'] = ship_test['Spa'].fillna(0.0)
ship_train.isnull().sum()
ship_train['VRDeck'].value_counts()
ship_train['VRDeck'] = ship_train['VRDeck'].fillna(0.0)
ship_test['VRDeck'] = ship_test['VRDeck'].fillna(0.0)
X = ship_train.drop(['Cabin', 'Name', 'Transported'], axis=1)
y = ship_train['Transported']
ship_train.dtypes
ship_train.head()
ship_train = pd.get_dummies(ship_train, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
ship_test = pd.get_dummies(ship_test, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
ship_train.dtypes
ship_train.head(5)
X = ship_train.drop(['PassengerId', 'Cabin', 'Name', 'Transported', 'CryoSleep_False', 'VIP_False'], axis=1)
y = ship_train['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_valid = le.transform(y_valid)
from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression(random_state=42)