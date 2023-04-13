import numpy as np
import pandas as pd
import os
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input0.head()
_input2.head()
_input1.shape
_input1.describe()
_input1.isnull().sum()
_input1.describe()
_input1['HomePlanet'].value_counts()
_input1['Transported'].value_counts()
_input1['Destination'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(_input1['Age'])
sns.barplot(data=_input1, x='VIP', y='Transported')
sns.barplot(data=_input1, x='CryoSleep', y='Transported')
sns.scatterplot(data=_input1, x='VIP', y='RoomService')
_input1.isnull().sum()
_input0.isnull().sum()
_input1['HomePlanet'].mode()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input1['CryoSleep'].mode()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False')
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('False')
_input1['Destination'].mode()
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
age_mean = _input1['Age'].mean()
age_mean
_input1['Age'] = _input1['Age'].fillna(round(age_mean))
_input0['Age'] = _input0['Age'].fillna(round(age_mean))
_input1['VIP'].mode()
_input1['VIP'] = _input1['VIP'].fillna('False')
_input0['VIP'] = _input0['VIP'].fillna('False')
_input1['RoomService'].mode()
_input1['RoomService'] = _input1['RoomService'].fillna(0.0)
_input0['RoomService'] = _input0['RoomService'].fillna(0.0)
_input1['FoodCourt'].mode()
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0.0)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0.0)
_input1['ShoppingMall'].value_counts()
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0.0)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0.0)
_input1['Spa'].value_counts()
_input1['Spa'] = _input1['Spa'].fillna(0.0)
_input0['Spa'] = _input0['Spa'].fillna(0.0)
_input1.isnull().sum()
_input1['VRDeck'].value_counts()
_input1['VRDeck'] = _input1['VRDeck'].fillna(0.0)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0.0)
X = _input1.drop(['Cabin', 'Name', 'Transported'], axis=1)
y = _input1['Transported']
_input1.dtypes
_input1.head()
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
_input0 = pd.get_dummies(_input0, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
_input1.dtypes
_input1.head(5)
X = _input1.drop(['PassengerId', 'Cabin', 'Name', 'Transported', 'CryoSleep_False', 'VIP_False'], axis=1)
y = _input1['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_valid = le.transform(y_valid)
from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression(random_state=42)