import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
train.info()
train['PassengerId'] = train['PassengerId'].str.split('_')
train['Group_num'] = train['PassengerId'].str[0]
train['Cabin'] = train['Cabin'].str.split('/')
train['Deck'] = train['Cabin'].str[0]
train['Room_num'] = train['Cabin'].str[1]
train['Side'] = train['Cabin'].str[2]
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
test['PassengerId'] = test['PassengerId'].str.split('_')
test['Group_num'] = test['PassengerId'].str[0]
test['Cabin'] = test['Cabin'].str.split('/')
test['Deck'] = test['Cabin'].str[0]
test['Room_num'] = test['Cabin'].str[1]
test['Side'] = test['Cabin'].str[2]
train['Group_num'].value_counts()
plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Deck', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Side', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train[train.Age < 19], x='Age', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train[(train.Age >= 19) & (train.Age < 35)], x='Age', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train[train.Age >= 35], x='Age', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='CryoSleep', hue='Transported')

plt.figure(figsize=(12, 8))
sns.countplot(data=train, x='HomePlanet', hue='Transported')

vips = train.VIP.value_counts()
vips = pd.DataFrame(vips)
vips
plt.figure(figsize=(12, 8))
plt.pie(vips['VIP'], labels=vips.index, explode=None)

plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='VIP', hue='Transported')

plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Destination', hue='Transported')

plt.figure(figsize=(12, 6))
sns.barplot(data=train, x='Transported', y='RoomService')

(fig, ax) = plt.subplots(2, 2, figsize=(12, 12))
sns.barplot(ax=ax[0, 0], data=train, x='Transported', y='VRDeck')
sns.barplot(ax=ax[0, 1], data=train, x='Transported', y='FoodCourt')
sns.barplot(ax=ax[1, 0], data=train, x='Transported', y='ShoppingMall')
sns.barplot(ax=ax[1, 1], data=train, x='Transported', y='Spa')

train.info()
test.info()
plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='HomePlanet', hue='VIP')

plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='HomePlanet', hue='Destination')

plt.figure(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True, cbar=False, linewidths=0.5)

train.drop(['PassengerId', 'Cabin', 'Room_num', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Cabin', 'Room_num', 'Name'], axis=1, inplace=True)
train['Group_num'] = train['Group_num'].astype('int64')
test['Group_num'] = test['Group_num'].astype('int64')
train['CryoSleep'] = train['CryoSleep'].astype(bool)
test['CryoSleep'] = test['CryoSleep'].astype(bool)
train['VIP'] = train['VIP'].astype(bool)
test['VIP'] = test['VIP'].astype(bool)
train.dropna(how='any', inplace=True)
train.info()
test.info()
test['HomePlanet'].fillna(value='Earth', inplace=True)
test['CryoSleep'].fillna(value=False, inplace=True)
test['Destination'].fillna(value=test.Destination.mode(), inplace=True)
test['Age'].fillna(value=test.Age.median(), inplace=True)
test['VIP'].fillna(value=False, inplace=True)
test['RoomService'].fillna(value=test.RoomService.median(), inplace=True)
test['FoodCourt'].fillna(value=test.FoodCourt.median(), inplace=True)
test['ShoppingMall'].fillna(value=test.ShoppingMall.median(), inplace=True)
test['Spa'].fillna(value=test.Spa.median(), inplace=True)
test['VRDeck'].fillna(value=test.VRDeck.median(), inplace=True)
test['Deck'].fillna(value='Other', inplace=True)
test['Side'].fillna(value='Unknown', inplace=True)
y_train = train['Transported']
X_train = train.drop('Transported', axis=1)
X_test = test.copy()
train.info()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.head()
print(X_test.shape)
print(X_train.shape)
X_test.columns
X_test.drop(['Side_Unknown', 'Deck_Other'], axis=1, inplace=True)
X_train.columns
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()