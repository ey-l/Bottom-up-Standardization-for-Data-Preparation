import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1['PassengerId'] = _input1['PassengerId'].str.split('_')
_input1['Group_num'] = _input1['PassengerId'].str[0]
_input1['Cabin'] = _input1['Cabin'].str.split('/')
_input1['Deck'] = _input1['Cabin'].str[0]
_input1['Room_num'] = _input1['Cabin'].str[1]
_input1['Side'] = _input1['Cabin'].str[2]
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0['PassengerId'] = _input0['PassengerId'].str.split('_')
_input0['Group_num'] = _input0['PassengerId'].str[0]
_input0['Cabin'] = _input0['Cabin'].str.split('/')
_input0['Deck'] = _input0['Cabin'].str[0]
_input0['Room_num'] = _input0['Cabin'].str[1]
_input0['Side'] = _input0['Cabin'].str[2]
_input1['Group_num'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='Deck', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='Side', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1[_input1.Age < 19], x='Age', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1[(_input1.Age >= 19) & (_input1.Age < 35)], x='Age', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1[_input1.Age >= 35], x='Age', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
plt.figure(figsize=(12, 8))
sns.countplot(data=_input1, x='HomePlanet', hue='Transported')
vips = _input1.VIP.value_counts()
vips = pd.DataFrame(vips)
vips
plt.figure(figsize=(12, 8))
plt.pie(vips['VIP'], labels=vips.index, explode=None)
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='VIP', hue='Transported')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='Destination', hue='Transported')
plt.figure(figsize=(12, 6))
sns.barplot(data=_input1, x='Transported', y='RoomService')
(fig, ax) = plt.subplots(2, 2, figsize=(12, 12))
sns.barplot(ax=ax[0, 0], data=_input1, x='Transported', y='VRDeck')
sns.barplot(ax=ax[0, 1], data=_input1, x='Transported', y='FoodCourt')
sns.barplot(ax=ax[1, 0], data=_input1, x='Transported', y='ShoppingMall')
sns.barplot(ax=ax[1, 1], data=_input1, x='Transported', y='Spa')
_input1.info()
_input0.info()
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='HomePlanet', hue='VIP')
plt.figure(figsize=(12, 6))
sns.countplot(data=_input1, x='HomePlanet', hue='Destination')
plt.figure(figsize=(10, 10))
sns.heatmap(_input1.corr(), annot=True, cbar=False, linewidths=0.5)
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Room_num', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Room_num', 'Name'], axis=1, inplace=False)
_input1['Group_num'] = _input1['Group_num'].astype('int64')
_input0['Group_num'] = _input0['Group_num'].astype('int64')
_input1['CryoSleep'] = _input1['CryoSleep'].astype(bool)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(bool)
_input1['VIP'] = _input1['VIP'].astype(bool)
_input0['VIP'] = _input0['VIP'].astype(bool)
_input1 = _input1.dropna(how='any', inplace=False)
_input1.info()
_input0.info()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(value='Earth', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(value=False, inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(value=_input0.Destination.mode(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(value=_input0.Age.median(), inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(value=False, inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(value=_input0.RoomService.median(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(value=_input0.FoodCourt.median(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(value=_input0.ShoppingMall.median(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(value=_input0.Spa.median(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(value=_input0.VRDeck.median(), inplace=False)
_input0['Deck'] = _input0['Deck'].fillna(value='Other', inplace=False)
_input0['Side'] = _input0['Side'].fillna(value='Unknown', inplace=False)
y_train = _input1['Transported']
X_train = _input1.drop('Transported', axis=1)
X_test = _input0.copy()
_input1.info()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.head()
print(X_test.shape)
print(X_train.shape)
X_test.columns
X_test = X_test.drop(['Side_Unknown', 'Deck_Other'], axis=1, inplace=False)
X_train.columns
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()