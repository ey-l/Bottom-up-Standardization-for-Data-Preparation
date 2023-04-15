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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(8)
df.shape
df['HomePlanet'].value_counts()
df['Destination'].value_counts()
df['CryoSleep'].value_counts()
df.isnull().sum()[df.isnull().sum() > 0]
df.shape
df['HomePlanet'] = df['HomePlanet'].map({'Earth': 0, 'Europa': 1, 'Mars': 2})
df['HomePlanet'].value_counts()
df['Destination'].value_counts()
df['Destination'] = df['Destination'].map({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
df['Destination'].value_counts()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['CryoSleep'] = encoder.fit_transform(df['CryoSleep'])
df['CryoSleep'].value_counts()
df['VIP'] = encoder.fit_transform(df['VIP'])
df['Transported'].value_counts()
df['Cabin'].value_counts()
df['Cabin'] = encoder.fit_transform(df['Cabin'])
df['Cabin'].value_counts()
df.isnull().sum()[df.isnull().sum() > 0]
df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)
df['Destination'].fillna(df['Destination'].mode()[0], inplace=True)
df['Age'].fillna(df['Age'].mode()[0], inplace=True)
df['RoomService'].fillna(df['RoomService'].mode()[0], inplace=True)
df['FoodCourt'].fillna(df['FoodCourt'].mode()[0], inplace=True)
df['ShoppingMall'].fillna(df['ShoppingMall'].mode()[0], inplace=True)
df['Spa'].fillna(df['Spa'].mode()[0], inplace=True)
df['VRDeck'].fillna(df['VRDeck'].mode()[0], inplace=True)
df.isnull().sum()[df.isnull().sum() > 0]
df.drop(['Name'], axis=1, inplace=True)
df.set_index('PassengerId', inplace=True)
df.head(10)
df['HomePlanet'] = df['HomePlanet'].astype(int)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr())
sns.barplot(x=df['Transported'], y=df['RoomService'])
train_features = df[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
train_target = df['Transported']
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df.head()
test_df.set_index('PassengerId', inplace=True)
test_df.isnull().sum()[test_df.isnull().sum() > 0]
test_df.shape
test_df['HomePlanet'].fillna(test_df['HomePlanet'].mode()[0], inplace=True)
test_df['Destination'].fillna(test_df['Destination'].mode()[0], inplace=True)
test_df['Age'].fillna(test_df['Age'].mode()[0], inplace=True)
test_df['RoomService'].fillna(test_df['RoomService'].mode()[0], inplace=True)
test_df['FoodCourt'].fillna(test_df['FoodCourt'].mode()[0], inplace=True)
test_df['ShoppingMall'].fillna(test_df['ShoppingMall'].mode()[0], inplace=True)
test_df['Spa'].fillna(test_df['Spa'].mode()[0], inplace=True)
test_df['VRDeck'].fillna(test_df['VRDeck'].mode()[0], inplace=True)
test_df['CryoSleep'].fillna(test_df['CryoSleep'].mode()[0], inplace=True)
test_df['Cabin'].fillna(test_df['Cabin'].mode()[0], inplace=True)
test_df['VIP'].fillna(test_df['VIP'].mode()[0], inplace=True)
test_df.drop(['Name'], axis=1, inplace=True)
test_df.shape
test_df['HomePlanet'] = test_df['HomePlanet'].map({'Earth': 0, 'Europa': 1, 'Mars': 2})
test_df['Destination'] = test_df['Destination'].map({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
test_df['CryoSleep'] = encoder.fit_transform(test_df['CryoSleep'])
test_df['VIP'] = encoder.fit_transform(test_df['VIP'])
test_df['Cabin'] = encoder.fit_transform(test_df['Cabin'])
test_df.head()
test_features = test_df[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', DecisionTreeClassifier())]
pipe = Pipeline(Input)