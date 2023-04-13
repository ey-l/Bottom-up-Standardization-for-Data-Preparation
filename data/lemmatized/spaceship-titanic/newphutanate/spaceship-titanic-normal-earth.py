import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train size :', _input1.shape)
print('Test size :', _input0.shape)
_input1.info()
_input1.describe()
_input1.head(5)
print('Missing Values Training Set')
_input1.isna().sum()
(f, ax) = plt.subplots(figsize=(6, 6))
_input1['Transported'].value_counts().plot.pie(autopct='%0.2f%%', explode=[0.05, 0.02], ax=ax)
plt.figure(figsize=(7, 4))
sns.countplot(data=_input1, x='HomePlanet')
_input1['HomePlanet'].value_counts()
plt.figure(figsize=(10, 4))
sns.histplot(data=_input1, x='Age')
plt.figure(figsize=(7, 4))
sns.countplot(data=_input1, x='Destination')
_input1['Destination'].value_counts()
plt.figure(figsize=(6, 4))
sns.countplot(data=_input1, x='CryoSleep')
_input1['CryoSleep'].value_counts()
plt.figure(figsize=(6, 4))
sns.countplot(data=_input1, x='VIP')
_input1['VIP'].value_counts()
plt.figure(figsize=(10, 4))
sns.countplot(data=_input1, x='Cabin')
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(10, 20))
for (i, var_name) in enumerate(exp_feats):
    ax = fig.add_subplot(5, 2, 2 * i + 1)
    sns.histplot(data=_input1, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    ax = fig.add_subplot(5, 2, 2 * i + 2)
    sns.histplot(data=_input1, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(var_name)
fig.tight_layout()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False, inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False, inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('Z/9999/Z', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('Z/9999/Z', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0], inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(False, inplace=False)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input1['TotalSpent'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
_input0['TotalSpent'] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
_input1[['CabinDeck', 'CabinNum', 'CabinSide']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['CabinDeck', 'CabinNum', 'CabinSide']] = _input0['Cabin'].str.split('/', expand=True)
plt.figure(figsize=(6, 4))
sns.countplot(data=_input1, x='CabinDeck')
plt.figure(figsize=(10, 4))
sns.countplot(data=_input1, x='CabinNum')
plt.figure(figsize=(10, 4))
sns.countplot(data=_input1, x='CabinSide')
_input1[['Group', 'GroupPos']] = _input1['PassengerId'].str.split('_', expand=True)
_input0[['Group', 'GroupPos']] = _input0['PassengerId'].str.split('_', expand=True)
plt.figure(figsize=(10, 4))
sns.countplot(data=_input1, x='Group')
plt.figure(figsize=(6, 4))
sns.countplot(data=_input1, x='GroupPos')
_input1['GroupPos'].value_counts()
_input1['AgeGroup'] = np.nan
_input1.loc[_input1['Age'] <= 5, 'AgeGroup'] = 0
_input1.loc[(_input1['Age'] > 5) & (_input1['Age'] <= 12), 'AgeGroup'] = 1
_input1.loc[(_input1['Age'] > 12) & (_input1['Age'] <= 17), 'AgeGroup'] = 2
_input1.loc[(_input1['Age'] > 17) & (_input1['Age'] <= 25), 'AgeGroup'] = 3
_input1.loc[(_input1['Age'] > 25) & (_input1['Age'] <= 30), 'AgeGroup'] = 4
_input1.loc[(_input1['Age'] > 30) & (_input1['Age'] <= 50), 'AgeGroup'] = 5
_input1.loc[_input1['Age'] > 50, 'AgeGroup'] = 6
_input0['AgeGroup'] = np.nan
_input0.loc[_input0['Age'] <= 5, 'AgeGroup'] = 0
_input0.loc[(_input0['Age'] > 5) & (_input0['Age'] <= 12), 'AgeGroup'] = 1
_input0.loc[(_input0['Age'] > 12) & (_input0['Age'] <= 17), 'AgeGroup'] = 2
_input0.loc[(_input0['Age'] > 17) & (_input0['Age'] <= 25), 'AgeGroup'] = 3
_input0.loc[(_input0['Age'] > 25) & (_input0['Age'] <= 30), 'AgeGroup'] = 4
_input0.loc[(_input0['Age'] > 30) & (_input0['Age'] <= 50), 'AgeGroup'] = 5
_input0.loc[_input0['Age'] > 50, 'AgeGroup'] = 6
sns.countplot(data=_input1, x='AgeGroup', hue='Transported')
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
_input1['CryoSleep'] = label.fit_transform(_input1['CryoSleep'])
_input0['CryoSleep'] = label.fit_transform(_input0['CryoSleep'])
_input1['VIP'] = label.fit_transform(_input1['VIP'])
_input0['VIP'] = label.fit_transform(_input0['VIP'])
_input1['Transported'] = label.fit_transform(_input1['Transported'])
from sklearn.preprocessing import OneHotEncoder
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide'])
_input1
_input0 = pd.get_dummies(_input0, columns=['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide'])
_input0
drop_col = ['Cabin', 'Name', 'CabinNum', 'Group', 'PassengerId']
_input1 = _input1.drop(columns=drop_col)
_input0 = _input0.drop(columns=drop_col)
_input1.columns
_input0.columns
X = _input1.drop(columns=['Transported']).values
y = _input1['Transported'].values
_input1['GroupPos'] = _input1['GroupPos'].astype(int)
_input0['GroupPos'] = _input0['GroupPos'].astype(int)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=101)
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
lg = LogisticRegression()