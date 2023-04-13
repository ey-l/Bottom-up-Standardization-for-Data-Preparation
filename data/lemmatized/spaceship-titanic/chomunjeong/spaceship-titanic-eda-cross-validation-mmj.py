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
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
len(_input0)
_input1.head(10)
_input0.head(10)
_input2.head(10)
_input1.info()
_input0.info()
_input2.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
missingno.matrix(_input2)
import missingno as msno
msno.bar(_input1)
import missingno as msno
msno.bar(_input0)
import missingno as msno
msno.bar(_input2)
train_df_del = _input1.drop(['Age', 'VRDeck', 'Cabin', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Transported'], axis=1)
train_df_del
for col in train_df_del:
    sns.countplot(x=col, hue='Transported', data=_input1, palette='Set2').set(title=col)
sns.countplot(x='HomePlanet', hue='Destination', data=_input1, palette='Set3')
import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.kdeplot(data=_input1, x='RoomService', hue='Transported', shade=True, ax=ax[0, 0])
sns.kdeplot(data=_input1, x='FoodCourt', hue='Transported', shade=True, ax=ax[0, 1])
sns.kdeplot(data=_input1, x='ShoppingMall', hue='Transported', shade=True, ax=ax[1, 0])
sns.kdeplot(data=_input1, x='Spa', hue='Transported', shade=True, ax=ax[1, 1])
sns.kdeplot(data=_input1, x='VRDeck', hue='Transported', shade=True, ax=ax[2, 0])
import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.histplot(data=_input1, x='RoomService', hue='Transported', bins=10, ax=ax[0, 0])
sns.histplot(data=_input1, x='FoodCourt', hue='Transported', bins=10, ax=ax[0, 1])
sns.histplot(data=_input1, x='ShoppingMall', hue='Transported', bins=10, ax=ax[1, 0])
sns.histplot(data=_input1, x='Spa', hue='Transported', bins=10, ax=ax[1, 1])
sns.histplot(data=_input1, x='Spa', hue='Transported', bins=10, ax=ax[2, 0])
_input1['Age']
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='Transported', bins=8, kde=True, ax=ax)
train_df_del = _input1.drop(['Age', 'VRDeck', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Cabin', 'VIP'], axis=1)
train_df_del
for col in train_df_del:
    sns.catplot(data=_input1, x=col, y='Age', hue='Transported', kind='violin', palette='pastel').set(title=col)
_input1.corr()
sns.heatmap(_input1.corr(), annot=True, cmap='coolwarm')
test_del = _input0.drop(['Age', 'VRDeck', 'Cabin', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt'], axis=1)
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
sns.kdeplot(data=_input0, x='RoomService', shade=True, ax=ax[0, 0])
sns.kdeplot(data=_input0, x='FoodCourt', shade=True, ax=ax[0, 1])
sns.kdeplot(data=_input0, x='ShoppingMall', shade=True, ax=ax[1, 0])
sns.kdeplot(data=_input0, x='Spa', shade=True, ax=ax[1, 1])
sns.kdeplot(data=_input0, x='VRDeck', shade=True, ax=ax[2, 0])
_input0['Age']
(fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.histplot(data=_input0, x='Age', bins=8, ax=ax[0])
sns.kdeplot(data=_input0, x='Age', ax=ax[1])
test_df_del = _input0.drop(['Age', 'VRDeck', 'RoomService', 'Spa', 'ShoppingMall', 'FoodCourt', 'Name', 'Cabin', 'VIP'], axis=1)
test_df_del
for col in test_df_del:
    sns.catplot(data=_input0, x=col, y='Age', hue='VIP', kind='violin', palette='pastel').set(title=col)
missingno.matrix(_input1)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(method='bfill', inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(method='bfill', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('G/734/S', inplace=False)
_input1
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
missingno.matrix(_input1)
_input1['HomePlanet'].value_counts()
_input1
missingno.matrix(_input0)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(method='bfill', inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(method='bfill', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('G/734/S', inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean(), inplace=False)
missingno.matrix(_input0)
_input1['Transported'].value_counts()
_input1['Transported'] = _input1['Transported'].astype(int)
_input1['Transported'].value_counts()
_input1['VIP'].value_counts()
_input1['VIP'] = _input1['VIP'].astype(int)
_input1['VIP'].value_counts()
_input0['VIP'] = _input0['VIP'].astype(int)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
_input1
_input0
data_df = pd.concat([_input1, _input0], axis=0)
missingno.matrix(data_df)
data_df
data_df = data_df.drop(['Name', 'Cabin'], axis=1, inplace=False)
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
train = data_std[0:len(_input1)]
test = data_std[len(_input1):]
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