import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import math
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.shape
df.head(20)
df.info()
print(df['PassengerId'].nunique())
print(df['Cabin'].nunique())
print(df['HomePlanet'].unique())
print(df['Destination'].unique())
df[df['HomePlanet'].isna()]
df[df['HomePlanet'].isna() & df['VIP'] == True]
sns.boxplot(data=df, x='VIP', y='Age', palette='mako')
df['Age'].mode()
df[df['Age'] == 24].shape
(fig, ax) = plt.subplots(2, 2, figsize=(20, 10))
sns.countplot(x='CryoSleep', data=df, ax=ax[0][0], palette='mako')
ax[0][0].set_title('Count of Passengers Being Put to Cryosleep and the ones that are not')
sns.boxplot(x='CryoSleep', y='Age', hue='Transported', data=df, ax=ax[0][1], palette='mako')
ax[0][1].set_title('Passengers being put to cryosleep with comparison to being transported and their age')
sns.countplot(x='VIP', data=df, ax=ax[1][0], palette='rocket')
ax[1][0].set_title('Count of VIP Passengers and the ones that are not')
sns.boxplot(x='VIP', y='Age', hue='Transported', data=df, ax=ax[1][1], palette='rocket')
ax[1][1].set_title('Passengers being put to cryosleep with comparison to being transported and their age')
df[df['Destination'].isna()]
df[df['Destination'].isna() & df['HomePlanet'].isna()]
sns.lmplot(x='RoomService', y='Age', hue='VIP', data=df, palette='mako')
sns.lmplot(x='RoomService', y='Age', hue='CryoSleep', data=df, palette='flare')
sns.lmplot(x='RoomService', y='Age', hue='Transported', data=df, palette='rocket')
sns.lmplot(x='FoodCourt', y='Age', hue='CryoSleep', data=df, palette='flare')
sns.lmplot(x='FoodCourt', y='Age', hue='VIP', data=df, palette='mako')
sns.lmplot(x='FoodCourt', y='Age', hue='Transported', data=df, palette='rocket')
sns.lmplot(x='ShoppingMall', y='Age', hue='CryoSleep', data=df, palette='flare')
sns.lmplot(x='ShoppingMall', y='Age', hue='VIP', data=df, palette='mako')
sns.lmplot(x='ShoppingMall', y='Age', hue='Transported', data=df, palette='rocket')
sns.lmplot(x='Spa', y='Age', hue='CryoSleep', data=df, palette='flare')
sns.lmplot(x='Spa', y='Age', hue='VIP', data=df, palette='mako')
sns.lmplot(x='Spa', y='Age', hue='Transported', data=df, palette='rocket')
sns.lmplot(x='VRDeck', y='Age', hue='CryoSleep', data=df, palette='flare')
sns.lmplot(x='VRDeck', y='Age', hue='VIP', data=df, palette='mako')
sns.lmplot(x='VRDeck', y='Age', hue='Transported', data=df, palette='rocket')
(fig, ax) = plt.subplots(3, figsize=(5, 20))
sns.countplot(data=df, x='HomePlanet', hue='Transported', ax=ax[0])
ax[0].set_title('Home Planet Distribution')
sns.countplot(data=df, x='Destination', hue='Transported', ax=ax[1], palette='mako')
ax[1].set_title('Destination Distribution')
sns.countplot(data=df, x='VIP', hue='Transported', ax=ax[2], palette='rocket')
ax[2].set_title('VIP Distribution')
df.info()
df.drop(df[df['HomePlanet'].isna()].index, inplace=True)
df.drop(df[df['Destination'].isna()].index, inplace=True)
df.drop('Cabin', inplace=True, axis=1)
df.drop('Name', inplace=True, axis=1)
df.drop(df[df['RoomService'] > 9500].index, inplace=True)
df.drop(df[df['FoodCourt'] > 25000].index, inplace=True)
df.drop(df[df['ShoppingMall'] > 20000].index, inplace=True)
df.drop(df[df['Spa'] > 17000].index, inplace=True)
df.drop(df[df['VRDeck'] > 19500].index, inplace=True)
df.drop('PassengerId', inplace=True, axis=1)
df.info()
df['CryoSleep'].fillna(False, inplace=True)
df['VIP'].fillna(False, inplace=True)
df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)
df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
df['Spa'].fillna(df['Spa'].mean(), inplace=True)
df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
newdf = df
newdf.info()
newdf = pd.get_dummies(newdf)
newdf
plt.subplots(figsize=(20, 20))
sns.heatmap(data=newdf.corr(), annot=True, cmap='Blues')
y = newdf.Transported
X = newdf[['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e']]
X
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=42, test_size=0.2, train_size=0.8)
lor = LogisticRegression(solver='liblinear', random_state=42)