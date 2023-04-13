import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(8)
_input1.shape
_input1['HomePlanet'].value_counts()
_input1['Destination'].value_counts()
_input1['CryoSleep'].value_counts()
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1.shape
_input1['HomePlanet'] = _input1['HomePlanet'].map({'Earth': 0, 'Europa': 1, 'Mars': 2})
_input1['HomePlanet'].value_counts()
_input1['Destination'].value_counts()
_input1['Destination'] = _input1['Destination'].map({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
_input1['Destination'].value_counts()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
_input1['CryoSleep'] = encoder.fit_transform(_input1['CryoSleep'])
_input1['CryoSleep'].value_counts()
_input1['VIP'] = encoder.fit_transform(_input1['VIP'])
_input1['Transported'].value_counts()
_input1['Cabin'].value_counts()
_input1['Cabin'] = encoder.fit_transform(_input1['Cabin'])
_input1['Cabin'].value_counts()
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mode()[0], inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mode()[0], inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mode()[0], inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mode()[0], inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mode()[0], inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mode()[0], inplace=False)
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input1 = _input1.set_index('PassengerId', inplace=False)
_input1.head(10)
_input1['HomePlanet'] = _input1['HomePlanet'].astype(int)
plt.figure(figsize=(10, 10))
sns.heatmap(_input1.corr())
sns.barplot(x=_input1['Transported'], y=_input1['RoomService'])
train_features = _input1[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
train_target = _input1['Transported']
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0 = _input0.set_index('PassengerId', inplace=False)
_input0.isnull().sum()[_input0.isnull().sum() > 0]
_input0.shape
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0], inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mode()[0], inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mode()[0], inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mode()[0], inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mode()[0], inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mode()[0], inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mode()[0], inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0], inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna(_input0['Cabin'].mode()[0], inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0], inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)
_input0.shape
_input0['HomePlanet'] = _input0['HomePlanet'].map({'Earth': 0, 'Europa': 1, 'Mars': 2})
_input0['Destination'] = _input0['Destination'].map({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
_input0['CryoSleep'] = encoder.fit_transform(_input0['CryoSleep'])
_input0['VIP'] = encoder.fit_transform(_input0['VIP'])
_input0['Cabin'] = encoder.fit_transform(_input0['Cabin'])
_input0.head()
test_features = _input0[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', DecisionTreeClassifier())]
pipe = Pipeline(Input)