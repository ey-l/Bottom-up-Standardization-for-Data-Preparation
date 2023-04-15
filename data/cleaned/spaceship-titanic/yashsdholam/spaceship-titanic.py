import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.head()
df_train.info()
df_test.info()
df_train.nunique()
df_test.nunique()
df_train.duplicated().sum()
df_test.duplicated().sum()
df_train.isnull().sum()
df_test.isnull().sum()
df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_train['HomePlanet'].fillna('Earth', inplace=True)
df_train['Destination'].fillna('TRAPPIST-1e', inplace=True)
df_train['VIP'].fillna(False, inplace=True)
df_train['CryoSleep'].fillna(False, inplace=True)
df_train['Cabin'].fillna('F/1/S', inplace=True)
df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_test['HomePlanet'].fillna('Earth', inplace=True)
df_test['Destination'].fillna('TRAPPIST-1e', inplace=True)
df_test['VIP'].fillna(False, inplace=True)
df_test['CryoSleep'].fillna(False, inplace=True)
df_test['Cabin'].fillna('F/1/S', inplace=True)
df_train[['Deck', 'Num', 'Side']] = df_train.Cabin.str.split('/', expand=True)
df_test[['Deck', 'Num', 'Side']] = df_test.Cabin.str.split('/', expand=True)
df_train['total_spent'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck']
df_test['total_spent'] = df_test['RoomService'] + df_test['FoodCourt'] + df_test['ShoppingMall'] + df_test['Spa'] + df_test['VRDeck']
df_train[['CryoSleep', 'VIP', 'Transported']] = df_train[['CryoSleep', 'VIP', 'Transported']].replace(False, 0)
df_train[['CryoSleep', 'VIP', 'Transported']] = df_train[['CryoSleep', 'VIP', 'Transported']].replace(True, 1)
df_test[['CryoSleep', 'VIP']] = df_test[['CryoSleep', 'VIP']].replace(False, 0)
df_test[['CryoSleep', 'VIP']] = df_test[['CryoSleep', 'VIP']].replace(True, 1)
df_train.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)
fig = px.histogram(df_train, x='HomePlanet', width=800, height=500)
fig.show()
fig = px.histogram(df_train, x='Destination', width=800, height=500)
fig.show()
fig = px.histogram(df_train, x='Age')
fig.show()
fig = px.pie(df_train, values='Transported', names='HomePlanet', width=800, height=500)
fig.show()
plt.figure(figsize=(8, 8))
sns.histplot(data=df_train, x='CryoSleep', hue='Transported')

col = ('HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck')
fig = plt.figure(figsize=(15.0, 12.0))
for (i, c) in enumerate(col):
    ax = plt.subplot(2, 3, 1 + i)
    ax = sns.countplot(data=df_train, x=c, hue='Transported')
plt.figure(figsize=(15, 10))
sns.heatmap(df_train.corr(), annot=True, cmap='Blues')

cols = df_train.select_dtypes(['object']).columns
cols
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
df_train[cols] = oe.fit_transform(df_train[cols])
df_train.head()
cols1 = ['HomePlanet', 'Destination', 'Deck', 'Num', 'Side']
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
df_test[cols1] = oe.fit_transform(df_test[cols1])
df_test.head()
features = df_train.drop(['PassengerId', 'Transported'], axis=1)
target = df_train['Transported']
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(features, target, random_state=1, test_size=0.2, stratify=target)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.5)