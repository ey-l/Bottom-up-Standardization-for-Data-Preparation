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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1
_input0
_input1.info()
_input0.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
num_cols = ['RoomService', 'FoodCourt', 'Age', 'Spa', 'VRDeck']
_input1[num_cols].corr()
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
sns.catplot(data=_input1, x='Age', y='Spa', kind='bar')
sns.catplot(data=_input1, x='Age', y='Spa', hue='HomePlanet', kind='bar')
sns.catplot(data=_input1, x='Age', y='ShoppingMall', kind='bar')
_input1.corr()
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='CryoSleep', bins=8, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Age', hue='CryoSleep', shade=True, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='VIP', bins=8, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Age', hue='VIP', shade=True, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='RoomService', hue='VIP', bins=40, ax=ax)
ax.set_xlim(0, 4000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='FoodCourt', hue='VIP', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='ShoppingMall', hue='VIP', bins=40, ax=ax)
ax.set_xlim(0, 6000)
sns.catplot(data=_input1, y='CryoSleep', col='HomePlanet', kind='bar')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 200)
missingno.matrix(_input1)
missingno.matrix(_input0)
data_df = pd.concat([_input1, _input0], axis=0)
data_df
missingno.matrix(data_df)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['FoodCourt'] = data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=False)
data_df['ShoppingMall'] = data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=False)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
data_df['HomePlanet'].value_counts()
data_df['Cabin'].value_counts()
data_df['Destination'].value_counts()
data_df['VIP'].value_counts()
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Mars', inplace=False)
data_df['CryoSleep'] = data_df['CryoSleep'].fillna('False', inplace=False)
data_df['Destination'] = data_df['Destination'].fillna('PSO J318.5-22', inplace=False)
data_df['VIP'] = data_df['VIP'].fillna('True', inplace=False)
data_df['Name'] = data_df['Name'].fillna('Smith', inplace=False)
data_df['Cabin'] = data_df['Cabin'].fillna('X000', inplace=False)
missingno.matrix(data_df)
data_df = data_df.drop(['Name', 'Cabin'], axis=1, inplace=False)
data_df
cat_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Transported']
data_oh = pd.get_dummies(data_df, columns=cat_cols)
data_oh
data_oh.describe()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
from sklearn.preprocessing import StandardScaler
data_std = data_oh.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_oh.describe()
data_std.describe()
from sklearn.preprocessing import MinMaxScaler
data_minmax = data_oh.copy()
scaler = MinMaxScaler()
data_minmax[num_cols] = scaler.fit_transform(data_minmax[num_cols])
data_oh.describe()
data_minmax.describe()
data_std
missingno.matrix(data_std)
train = data_std[0:len(_input1)]
test = data_std[len(_input1):]
train.info()
test.info()
train
y = train['Transported_True']
X = train.drop('Transported_True', axis=1)
X_test = test.drop('Transported_True', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()