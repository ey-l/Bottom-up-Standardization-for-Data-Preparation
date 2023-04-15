import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')


train_df.info()
train_df['CryoSleep'].isnull().values.sum() / len(train_df)
test_df.info()
test_df['FoodCourt'].isnull().values.sum() / len(test_df)
missingno.matrix(train_df)
missingno.matrix(test_df)
data_df = pd.concat([train_df, test_df], axis=0)
data_df
missingno.matrix(data_df)
data_df.info()
train_df['Transported'] = train_df['Transported'].astype(int)
train_df
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
train_df[num_cols].corr()
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap='coolwarm')
sns.kdeplot(data=train_df, x='Age', hue='Transported', shade=True)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='RoomService', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-500, 2500)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='FoodCourt', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1500, 5000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='ShoppingMall', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-500, 3000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='Spa', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1000, 1500)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='VRDeck', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-1000, 1500)

sns.catplot(data=train_df, x='CryoSleep', y='Transported', kind='bar')
sns.catplot(data=train_df, x='HomePlanet', y='Transported', kind='bar')
train_df['Cabin'].fillna('X', inplace=True)
train_df['Cabin'].value_counts()
train_df['Deck'] = train_df['Cabin'].str[0]
train_df['Deck'].value_counts()
train_df.groupby('Deck').count()
sns.catplot(data=train_df, x='Deck', y='Transported', kind='bar')
sns.catplot(data=train_df, x='Destination', y='Transported', kind='bar')
sns.catplot(data=train_df, x='VIP', y='Transported', kind='bar')