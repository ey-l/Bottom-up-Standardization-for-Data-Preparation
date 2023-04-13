import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1.info()
_input1['CryoSleep'].isnull().values.sum() / len(_input1)
_input0.info()
_input0['FoodCourt'].isnull().values.sum() / len(_input0)
missingno.matrix(_input1)
missingno.matrix(_input0)
data_df = pd.concat([_input1, _input0], axis=0)
data_df
missingno.matrix(data_df)
data_df.info()
_input1['Transported'] = _input1['Transported'].astype(int)
_input1
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
_input1[num_cols].corr()
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
sns.kdeplot(data=_input1, x='Age', hue='Transported', shade=True)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='RoomService', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-500, 2500)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='FoodCourt', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1500, 5000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='ShoppingMall', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-500, 3000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Spa', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1000, 1500)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='VRDeck', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1000, 1500)
sns.catplot(data=_input1, x='CryoSleep', y='Transported', kind='bar')
sns.catplot(data=_input1, x='HomePlanet', y='Transported', kind='bar')
_input1['Cabin'] = _input1['Cabin'].fillna('X', inplace=False)
_input1['Cabin'].value_counts()
_input1['Deck'] = _input1['Cabin'].str[0]
_input1['Deck'].value_counts()
_input1.groupby('Deck').count()
sns.catplot(data=_input1, x='Deck', y='Transported', kind='bar')
sns.catplot(data=_input1, x='Destination', y='Transported', kind='bar')
sns.catplot(data=_input1, x='VIP', y='Transported', kind='bar')