import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
sample_df = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
len(test_df)
train_df.head(10)
test_df.head(10)
sample_df.head(10)
train_df.info()
test_df.info()
sample_df.info()
missingno.matrix(train_df)
missingno.matrix(test_df)
missingno.matrix(sample_df)
import missingno as msno
msno.bar(train_df)
import missingno as msno
msno.bar(test_df)
import missingno as msno
msno.bar(sample_df)
train_df_del = train_df.drop(['Age', 'VRDeck', 'Cabin', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Transported'], axis=1)
train_df_del
for col in train_df_del:
    sns.countplot(x=col, hue='Transported', data=train_df, palette='Set2').set(title=col)

sns.countplot(x='HomePlanet', hue='Destination', data=train_df, palette='Set3')

import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.kdeplot(data=train_df, x='RoomService', hue='Transported', shade=True, ax=ax[0, 0])
sns.kdeplot(data=train_df, x='FoodCourt', hue='Transported', shade=True, ax=ax[0, 1])
sns.kdeplot(data=train_df, x='ShoppingMall', hue='Transported', shade=True, ax=ax[1, 0])
sns.kdeplot(data=train_df, x='Spa', hue='Transported', shade=True, ax=ax[1, 1])
sns.kdeplot(data=train_df, x='VRDeck', hue='Transported', shade=True, ax=ax[2, 0])
import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.histplot(data=train_df, x='RoomService', hue='Transported', bins=10, ax=ax[0, 0])
sns.histplot(data=train_df, x='FoodCourt', hue='Transported', bins=10, ax=ax[0, 1])
sns.histplot(data=train_df, x='ShoppingMall', hue='Transported', bins=10, ax=ax[1, 0])
sns.histplot(data=train_df, x='Spa', hue='Transported', bins=10, ax=ax[1, 1])
sns.histplot(data=train_df, x='Spa', hue='Transported', bins=10, ax=ax[2, 0])
train_df['Age']
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='Age', hue='Transported', bins=8, kde=True, ax=ax)

train_df_del = train_df.drop(['Age', 'VRDeck', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Cabin', 'VIP'], axis=1)
train_df_del
for col in train_df_del:
    sns.catplot(data=train_df, x=col, y='Age', hue='Transported', kind='violin', palette='pastel').set(title=col)

train_df.corr()
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
test_del = test_df.drop(['Age', 'VRDeck', 'Cabin', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt'], axis=1)
test_del
import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
sns.countplot(x='HomePlanet', hue='HomePlanet', data=test_del, palette='Set2', ax=ax[0, 0])
sns.countplot(x='CryoSleep', hue='CryoSleep', data=test_del, palette='Set2', ax=ax[0, 1])
sns.countplot(x='Destination', hue='Destination', data=test_del, palette='Set2', ax=ax[1, 0])
sns.countplot(x='VIP', hue='VIP', data=test_del, palette='Set2', ax=ax[1, 1])

import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.kdeplot(data=test_df, x='RoomService', shade=True, ax=ax[0, 0])
sns.kdeplot(data=test_df, x='FoodCourt', shade=True, ax=ax[0, 1])
sns.kdeplot(data=test_df, x='ShoppingMall', shade=True, ax=ax[1, 0])
sns.kdeplot(data=test_df, x='Spa', shade=True, ax=ax[1, 1])
sns.kdeplot(data=test_df, x='VRDeck', shade=True, ax=ax[2, 0])
test_df['Age']
(fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.histplot(data=test_df, x='Age', bins=8, ax=ax[0])
sns.kdeplot(data=test_df, x='Age', ax=ax[1])

test_df_del = test_df.drop(['Age', 'VRDeck', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Cabin', 'VIP'], axis=1)
test_df_del
for col in test_df_del:
    sns.catplot(data=test_df, x=col, y='Age', hue='VIP', kind='violin', palette='pastel').set(title=col)

missingno.matrix(train_df)
train_df['HomePlanet'].fillna('Earth', inplace=True)
train_df['CryoSleep'].fillna(method='bfill', inplace=True)
train_df['VIP'].fillna(method='bfill', inplace=True)
train_df['Destination'].fillna('TRAPPIST-1e', inplace=True)
train_df['Cabin'].fillna('G/734/S', inplace=True)
train_df
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df['RoomService'].fillna(train_df['RoomService'].mean(), inplace=True)
train_df['FoodCourt'].fillna(train_df['FoodCourt'].mean(), inplace=True)
train_df['ShoppingMall'].fillna(train_df['ShoppingMall'].mean(), inplace=True)
train_df['Spa'].fillna(train_df['Spa'].mean(), inplace=True)
train_df['VRDeck'].fillna(train_df['VRDeck'].mean(), inplace=True)
missingno.matrix(train_df)
train_df['HomePlanet'].value_counts()
train_df
missingno.matrix(test_df)
test_df['HomePlanet'].fillna('Earth', inplace=True)
test_df['CryoSleep'].fillna(method='bfill', inplace=True)
test_df['VIP'].fillna(method='bfill', inplace=True)
test_df['Destination'].fillna('TRAPPIST-1e', inplace=True)
test_df['Cabin'].fillna('G/734/S', inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['RoomService'].fillna(test_df['RoomService'].mean(), inplace=True)
test_df['FoodCourt'].fillna(test_df['FoodCourt'].mean(), inplace=True)
test_df['ShoppingMall'].fillna(test_df['ShoppingMall'].mean(), inplace=True)
test_df['Spa'].fillna(test_df['Spa'].mean(), inplace=True)
test_df['VRDeck'].fillna(test_df['VRDeck'].mean(), inplace=True)
missingno.matrix(test_df)
train_df['Transported'].value_counts()
train_df['Transported'] = train_df['Transported'].astype(int)
train_df['Transported'].value_counts()
train_df['VIP'].value_counts()
train_df['VIP'] = train_df['VIP'].astype(int)
train_df['VIP'].value_counts()
test_df['VIP'] = test_df['VIP'].astype(int)
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)
train_df
test_df
data_df = pd.concat([train_df, test_df], axis=0)
missingno.matrix(data_df)
data_df
data_df.drop(['Name', 'Cabin'], axis=1, inplace=True)
data_df
cat_cols = ['HomePlanet', 'Destination']
data_en = pd.get_dummies(data_df, columns=cat_cols)
data_en
missingno.matrix(data_en)
data_en.describe()
num_cols = ['Age', 'VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService']
from sklearn.preprocessing import StandardScaler
data_std = data_en.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_en.describe()
data_std
missingno.matrix(data_std)
train = data_std[0:len(train_df)]
test = data_std[len(train_df):]
train.info()
test.info()
train
y = train['Transported']
x = train.drop('Transported', axis=1)
x_test = test.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(x, y, test_size=0.2, random_state=56)
y
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()