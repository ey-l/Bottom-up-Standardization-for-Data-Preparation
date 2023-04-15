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
train_df
test_df
train_df.info()
test_df.info()
missingno.matrix(train_df)
missingno.matrix(test_df)
train_df.info()
train_df = train_df.astype({'Transported': 'int'})
train_df
train_df.info()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap='coolwarm')
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap='coolwarm')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='RoomService', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='Spa', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='ShoppingMall', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='Age', hue='Transported', bins=8, ax=ax)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='Age', hue='Transported', shade=True, ax=ax)

sns.heatmap(train_df[num_cols].corr(), annot=True, cmap='coolwarm')
train_df
train_df.drop(['ShoppingMall', 'FoodCourt', 'Cabin', 'Name'], axis=1, inplace=True)
test_df.drop(['ShoppingMall', 'FoodCourt', 'Cabin', 'Name'], axis=1, inplace=True)
train_df
test_df
data_df = pd.concat([train_df, test_df], axis=0)
test_df
data_df
missingno.matrix(data_df)
data_df.info()
data_df['HomePlanet'].value_counts()
data_df['HomePlanet'].fillna('Earth', inplace=True)
train_df['CryoSleep'].value_counts()
data_df['CryoSleep'].fillna('False', inplace=True)
train_df['Destination'].value_counts()
data_df['Destination'].fillna('False', inplace=True)
train_df['VIP'].value_counts()
data_df['VIP'].fillna('False', inplace=True)
missingno.matrix(data_df)
data_df['Age'].fillna(data_df['Age'].mean(), inplace=True)
data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=True)
data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=True)
data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=True)
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
train = data_std[0:len(train_df)]
test = data_std[len(train_df):]
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