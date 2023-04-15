import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
spaceship_titanic_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
spaceship_titanic_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
spaceship_titanic_sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
spaceship_titanic_train.head()
spaceship_titanic_train.isnull().sum()
spaceship_titanic_train.info()
plt.figure(figsize=(15, 8))
x = spaceship_titanic_train['Transported'].value_counts()
mylabel = ['Not Transported (0)', 'Transported(1)']
colors = ['#f4acb7', '#9d8189']
plt.pie(x, labels=mylabel, autopct='%1.1f%%', startangle=15, shadow=True, colors=colors)
plt.axis('equal')

plt.figure(figsize=(15, 8))
hue_color = {0: '#012a4a', 1: '#2c7da0'}
ax = sns.countplot(data=spaceship_titanic_train, x='HomePlanet', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])

plt.figure(figsize=(15, 8))
hue_color = {0: '#E63946', 1: '#F1FAEE'}
ax = sns.countplot(data=spaceship_titanic_train, x='CryoSleep', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])

plt.figure(figsize=(15, 8))
hue_color = {0: '#8D99AE', 1: '#ef233c'}
ax = sns.countplot(data=spaceship_titanic_train, x='Destination', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])

plt.figure(figsize=(15, 8))
sns.countplot(x=spaceship_titanic_train['Transported'], hue=pd.cut(spaceship_titanic_train['Age'], 5))
spaceship_titanic_train.dropna(inplace=True)
spaceship_titanic_train.drop(['Cabin', 'PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
spaceship_titanic_test.drop(['Cabin', 'PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
spaceship_titanic_train['CryoSleep'].replace({'False': 0, 'True': 1}, inplace=True)
spaceship_titanic_train['VIP'].replace({'False': 0, 'True': 1}, inplace=True)
spaceship_titanic_train['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3}, inplace=True)
spaceship_titanic_train['Destination'].replace({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}, inplace=True)
spaceship_titanic_test['CryoSleep'] = spaceship_titanic_test['CryoSleep'].fillna('False')
spaceship_titanic_test['VIP'] = spaceship_titanic_test['VIP'].fillna('False')
spaceship_titanic_test['HomePlanet'] = spaceship_titanic_test['HomePlanet'].fillna('Earth')
spaceship_titanic_test['Destination'] = spaceship_titanic_test['Destination'].fillna('TRAPPIST-1e')
spaceship_titanic_test['Age'] = spaceship_titanic_test['Age'].fillna(spaceship_titanic_test['Age'].mean())
spaceship_titanic_test['CryoSleep'].replace({'False': 0, 'True': 1}, inplace=True)
spaceship_titanic_test['VIP'].replace({'False': 0, 'True': 1}, inplace=True)
spaceship_titanic_test['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3}, inplace=True)
spaceship_titanic_test['Destination'].replace({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}, inplace=True)
X = spaceship_titanic_train.drop(['Transported'], axis='columns')
y = spaceship_titanic_train['Transported']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()