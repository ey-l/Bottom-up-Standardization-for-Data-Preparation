import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train size :', train.shape)
print('Test size :', test.shape)
train.info()
train.describe()
train.head(5)
print('Missing Values Training Set')
train.isna().sum()
(f, ax) = plt.subplots(figsize=(6, 6))
train['Transported'].value_counts().plot.pie(autopct='%0.2f%%', explode=[0.05, 0.02], ax=ax)
plt.figure(figsize=(7, 4))
sns.countplot(data=train, x='HomePlanet')
train['HomePlanet'].value_counts()
plt.figure(figsize=(10, 4))
sns.histplot(data=train, x='Age')
plt.figure(figsize=(7, 4))
sns.countplot(data=train, x='Destination')
train['Destination'].value_counts()
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='CryoSleep')
train['CryoSleep'].value_counts()
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='VIP')
train['VIP'].value_counts()
plt.figure(figsize=(10, 4))
sns.countplot(data=train, x='Cabin')
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(10, 20))
for (i, var_name) in enumerate(exp_feats):
    ax = fig.add_subplot(5, 2, 2 * i + 1)
    sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    ax = fig.add_subplot(5, 2, 2 * i + 2)
    sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(var_name)
fig.tight_layout()

train['HomePlanet'].fillna(train['HomePlanet'].mode()[0], inplace=True)
test['HomePlanet'].fillna(test['HomePlanet'].mode()[0], inplace=True)
train['CryoSleep'].fillna(False, inplace=True)
test['CryoSleep'].fillna(False, inplace=True)
train['Cabin'].fillna('Z/9999/Z', inplace=True)
test['Cabin'].fillna('Z/9999/Z', inplace=True)
train['Destination'].fillna(train['Destination'].mode()[0], inplace=True)
test['Destination'].fillna(test['Destination'].mode()[0], inplace=True)
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['VIP'].fillna(False, inplace=True)
test['VIP'].fillna(False, inplace=True)
train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
train['TotalSpent'] = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test['TotalSpent'] = test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
train[['CabinDeck', 'CabinNum', 'CabinSide']] = train['Cabin'].str.split('/', expand=True)
test[['CabinDeck', 'CabinNum', 'CabinSide']] = test['Cabin'].str.split('/', expand=True)
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='CabinDeck')
plt.figure(figsize=(10, 4))
sns.countplot(data=train, x='CabinNum')
plt.figure(figsize=(10, 4))
sns.countplot(data=train, x='CabinSide')
train[['Group', 'GroupPos']] = train['PassengerId'].str.split('_', expand=True)
test[['Group', 'GroupPos']] = test['PassengerId'].str.split('_', expand=True)
plt.figure(figsize=(10, 4))
sns.countplot(data=train, x='Group')
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='GroupPos')
train['GroupPos'].value_counts()
train['AgeGroup'] = np.nan
train.loc[train['Age'] <= 5, 'AgeGroup'] = 0
train.loc[(train['Age'] > 5) & (train['Age'] <= 12), 'AgeGroup'] = 1
train.loc[(train['Age'] > 12) & (train['Age'] <= 17), 'AgeGroup'] = 2
train.loc[(train['Age'] > 17) & (train['Age'] <= 25), 'AgeGroup'] = 3
train.loc[(train['Age'] > 25) & (train['Age'] <= 30), 'AgeGroup'] = 4
train.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'AgeGroup'] = 5
train.loc[train['Age'] > 50, 'AgeGroup'] = 6
test['AgeGroup'] = np.nan
test.loc[test['Age'] <= 5, 'AgeGroup'] = 0
test.loc[(test['Age'] > 5) & (test['Age'] <= 12), 'AgeGroup'] = 1
test.loc[(test['Age'] > 12) & (test['Age'] <= 17), 'AgeGroup'] = 2
test.loc[(test['Age'] > 17) & (test['Age'] <= 25), 'AgeGroup'] = 3
test.loc[(test['Age'] > 25) & (test['Age'] <= 30), 'AgeGroup'] = 4
test.loc[(test['Age'] > 30) & (test['Age'] <= 50), 'AgeGroup'] = 5
test.loc[test['Age'] > 50, 'AgeGroup'] = 6
sns.countplot(data=train, x='AgeGroup', hue='Transported')
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train['CryoSleep'] = label.fit_transform(train['CryoSleep'])
test['CryoSleep'] = label.fit_transform(test['CryoSleep'])
train['VIP'] = label.fit_transform(train['VIP'])
test['VIP'] = label.fit_transform(test['VIP'])
train['Transported'] = label.fit_transform(train['Transported'])
from sklearn.preprocessing import OneHotEncoder
train = pd.get_dummies(train, columns=['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide'])
train
test = pd.get_dummies(test, columns=['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide'])
test
drop_col = ['Cabin', 'Name', 'CabinNum', 'Group', 'PassengerId']
train = train.drop(columns=drop_col)
test = test.drop(columns=drop_col)
train.columns
test.columns
X = train.drop(columns=['Transported']).values
y = train['Transported'].values
train['GroupPos'] = train['GroupPos'].astype(int)
test['GroupPos'] = test['GroupPos'].astype(int)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=101)
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
lg = LogisticRegression()