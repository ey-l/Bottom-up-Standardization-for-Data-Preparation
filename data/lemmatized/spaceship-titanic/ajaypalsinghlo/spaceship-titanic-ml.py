import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.isnull().sum()
_input1.info()
plt.figure(figsize=(15, 8))
x = _input1['Transported'].value_counts()
mylabel = ['Not Transported (0)', 'Transported(1)']
colors = ['#f4acb7', '#9d8189']
plt.pie(x, labels=mylabel, autopct='%1.1f%%', startangle=15, shadow=True, colors=colors)
plt.axis('equal')
plt.figure(figsize=(15, 8))
hue_color = {0: '#012a4a', 1: '#2c7da0'}
ax = sns.countplot(data=_input1, x='HomePlanet', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])
plt.figure(figsize=(15, 8))
hue_color = {0: '#E63946', 1: '#F1FAEE'}
ax = sns.countplot(data=_input1, x='CryoSleep', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])
plt.figure(figsize=(15, 8))
hue_color = {0: '#8D99AE', 1: '#ef233c'}
ax = sns.countplot(data=_input1, x='Destination', hue='Transported', palette=hue_color)
plt.legend(['Percentage not Transported', 'Percentage of Transported'])
plt.figure(figsize=(15, 8))
sns.countplot(x=_input1['Transported'], hue=pd.cut(_input1['Age'], 5))
_input1 = _input1.dropna(inplace=False)
_input1 = _input1.drop(['Cabin', 'PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Cabin', 'PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace({'False': 0, 'True': 1}, inplace=False)
_input1['VIP'] = _input1['VIP'].replace({'False': 0, 'True': 1}, inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3}, inplace=False)
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}, inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('False')
_input0['VIP'] = _input0['VIP'].fillna('False')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0['CryoSleep'] = _input0['CryoSleep'].replace({'False': 0, 'True': 1}, inplace=False)
_input0['VIP'] = _input0['VIP'].replace({'False': 0, 'True': 1}, inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3}, inplace=False)
_input0['Destination'] = _input0['Destination'].replace({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}, inplace=False)
X = _input1.drop(['Transported'], axis='columns')
y = _input1['Transported']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()