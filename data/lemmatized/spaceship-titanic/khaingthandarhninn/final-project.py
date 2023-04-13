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
_input1.info()
_input0.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
_input1[_input1['FoodCourt'].isnull()]
_input0[_input0['CryoSleep'].isnull()]
_input1.groupby(by='Age').count()
_input1.groupby(by='Transported').count()
_input1.head(5)
data_df = pd.concat([_input1, _input0], axis=0)
data_df
missingno.matrix(data_df)
data_df.info()
data_df.groupby(by='HomePlanet').count()
data_df.groupby(by='Destination').count()
data_df.groupby(by=['VIP', 'HomePlanet']).count()
sns.catplot(data=_input1, x='HomePlanet', y='Transported', hue='Destination', kind='bar')
sns.catplot(data=_input1, x='VIP', y='Transported', hue='Destination', col='HomePlanet', kind='bar')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Age', hue='Transported', shade=True, ax=ax)
ax.set_xlim(-100, 300)
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1[num_cols].corr()
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
missingno.matrix(_input1)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['Cabin'] = data_df['Cabin'].fillna('X', inplace=False)
data_df['Cabin'].value_counts()
data_df['Deck'] = data_df['Cabin'].str[0]
data_df['Deck'].count()
data_df.groupby(by='Deck').mean()
data_df.groupby(by=['Transported', 'Deck']).count()
sns.catplot(data=data_df, x='Deck', y='Transported', col='HomePlanet', kind='bar')
data_df['HomePlanet'].value_counts()
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Earth', inplace=False)
data_df['Destination'].value_counts()
data_df['Destination'] = data_df['Destination'].fillna('TRAPPIST-1e', inplace=False)
data_df['Name'] = data_df['Name'].fillna('XXX', inplace=False)
missingno.matrix(data_df)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
data_df['FoodCourt'] = data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=False)
data_df['ShoppingMall'] = data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['CryoSleep'].value_counts()
data_df['CryoSleep'] = data_df['CryoSleep'].fillna('False', inplace=False)
data_df['VIP'].value_counts()
data_df['VIP'] = data_df['VIP'].fillna('False', inplace=False)
data_df['Entertainment'] = data_df['RoomService'] + data_df['VRDeck'] + data_df['Spa'] + data_df['FoodCourt'] + data_df['ShoppingMall']
missingno.matrix(data_df)
data_df
data_df = data_df.drop('Entertainment', axis=1)
data_df = data_df.drop('Deck', axis=1)
data_df = data_df.drop('Cabin', axis=1)
data_df = data_df.drop('Name', axis=1)
data_df.info()
for data in data_df:
    data_df['HomePlanet'] = data_df['HomePlanet'].astype('category').cat.codes
    data_df['Destination'] = data_df['Destination'].astype('category').cat.codes
    data_df['CryoSleep'] = data_df['CryoSleep'].astype('category').cat.codes
    data_df['VIP'] = data_df['VIP'].astype('category').cat.codes
data_df
from sklearn.preprocessing import StandardScaler
data_df['Entertainment'] = data_df['RoomService'] + data_df['Spa'] + data_df['ShoppingMall'] + data_df['VRDeck'] + data_df['FoodCourt']
num_cols = ['Entertainment']
data_df = data_df.drop(['RoomService', 'Spa', 'ShoppingMall', 'VRDeck', 'FoodCourt'], axis=1, inplace=False)
data_df.describe()
data_std = data_df.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_std.describe()
data_std
train = data_std[0:len(_input1)]
test = data_std[len(_input1):]
train.info()
train
test
y = train['Transported']
X = train.drop('Transported', axis=1)
y = y.astype('category').cat.codes
X_test = test.drop('Transported', axis=1)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()