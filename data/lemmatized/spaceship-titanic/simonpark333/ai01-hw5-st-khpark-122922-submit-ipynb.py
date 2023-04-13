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
_input1.info()
_input1 = _input1.astype({'Transported': 'int'})
_input1
_input1.info()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='RoomService', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Spa', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='ShoppingMall', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='Transported', bins=8, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Age', hue='Transported', shade=True, ax=ax)
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
_input1
_input1 = _input1.drop(['ShoppingMall', 'FoodCourt', 'Cabin', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['ShoppingMall', 'FoodCourt', 'Cabin', 'Name'], axis=1, inplace=False)
_input1
_input0
data_df = pd.concat([_input1, _input0], axis=0)
_input0
data_df
missingno.matrix(data_df)
data_df.info()
data_df['HomePlanet'].value_counts()
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Earth', inplace=False)
_input1['CryoSleep'].value_counts()
data_df['CryoSleep'] = data_df['CryoSleep'].fillna('False', inplace=False)
_input1['Destination'].value_counts()
data_df['Destination'] = data_df['Destination'].fillna('False', inplace=False)
_input1['VIP'].value_counts()
data_df['VIP'] = data_df['VIP'].fillna('False', inplace=False)
missingno.matrix(data_df)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
missingno.matrix(data_df)
data_df.info()
data_df.dtypes
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
data_oh = pd.get_dummies(data_df, columns=cat_cols)
pd.get_dummies(data_df['VIP'], columns=['VIP'])
data_df.CryoSleep.loc[data_df.CryoSleep == 'False'] = False
data_df.VIP.loc[data_df.VIP == 'False'] = False
data_oh
data_df.describe()
data_oh = pd.get_dummies(data_df, columns=cat_cols)
data_oh.info()
data_oh.describe()
num_cols = ['Age', 'RoomService', 'Spa', 'VRDeck']
from sklearn.preprocessing import StandardScaler
data_std = data_oh.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_oh.describe()
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
y = train['Transported']
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
X_test
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=56)
X_train
X_val
y_train
y_val
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()