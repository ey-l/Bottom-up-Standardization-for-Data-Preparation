import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.describe()
_input1.dtypes
for i in _input1.columns:
    print(i, _input1[i].isna().sum())
_input1.dtypes
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = _input1[categorical_columns].mode().iloc[0]
_input1[categorical_columns] = _input1[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = _input1[numerical_columns].median()
_input1[numerical_columns] = _input1[numerical_columns].fillna(median)
for i in _input1.columns:
    print(i, _input1[i].isna().sum())
_input1 = _input1.drop(columns=['Name'])
import seaborn as sns
import matplotlib.pyplot as plt
corr = _input1.corr()
(fig, ax) = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, ax=ax)
_input1['Expenses'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1 = _input1.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
_input1.head()
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
_input1['Age Group'] = pd.cut(_input1['Age'], bins=bins, labels=labels, right=False)
_input1.head()
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['Age Group'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['HomePlanet'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['Transported'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['CryoSleep'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
string = _input1['Cabin'].str.split('/')
_input1['Deck'] = string.map(lambda string: string[0])
_input1['Number'] = string.map(lambda string: string[1])
_input1['Side'] = string.map(lambda string: string[2])
string = _input1['PassengerId'].str.split('_')
_input1['Group'] = string.map(lambda string: string[0])
_input1['Psngr_Num'] = string.map(lambda string: string[1])
_input1 = _input1.drop(columns=['Cabin', 'PassengerId'])
_input1
_input1['Deck'].unique()
_input1['Side'].unique()
_input1['Number'].unique()
_input1['Psngr_Num'].unique()
_input1['Group'].unique()
_input1 = _input1.drop(columns=['Number', 'Group'])
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['Deck'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = _input1['Side'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
_input1['Transported'] = encoder.fit_transform(_input1['Transported'])
_input1['CryoSleep'] = encoder.fit_transform(_input1['CryoSleep'])
_input1['HomePlanet'] = encoder.fit_transform(_input1['HomePlanet'])
_input1['Age Group'] = encoder.fit_transform(_input1['Age Group'])
_input1['Destination'] = encoder.fit_transform(_input1['Destination'])
_input1['VIP'] = encoder.fit_transform(_input1['VIP'])
_input1['Side'] = encoder.fit_transform(_input1['Side'])
_input1['Deck'] = encoder.fit_transform(_input1['Deck'])
_input1.head()
corr = _input1.corr()
(fig, ax) = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, ax=ax)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
_input1[['Age', 'Expenses']] = ss.fit_transform(_input1[['Age', 'Expenses']])
_input1.head()
X_Train = _input1.drop('Transported', axis=1)
Y_Train = _input1['Transported']
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
(X_train, X_test, y_train, y_test) = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=0)
logreg = LogisticRegression()