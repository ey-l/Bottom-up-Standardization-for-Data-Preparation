import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import string
import random
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Shape of train.csv : ', _input1.shape)
print('Shape of test.csv : ', _input0.shape)
_input1.head()
_input0.head()
_input1.info()
_input1.isnull().sum()
_input0.info()
_input0.isnull().sum()
print(_input1['Age'].median())
print(_input0['Age'].median())
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
column = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    print(column[i].value_counts())
column = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
    _input0[i] = _input0[i].fillna(_input0[i].mode()[0], inplace=False)
Others = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Others:
    print(_input1[i].median())
Others = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Others:
    _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
    _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
_input1 = _input1.drop(columns=['Cabin', 'Name'], axis=1)
_input0 = _input0.drop(columns=['Cabin', 'Name'], axis=1)
print('Shape of train.csv : ', _input1.shape)
print('Shape of test.csv : ', _input0.shape)
_input1[['CryoSleep', 'VIP']] = _input1[['CryoSleep', 'VIP']].astype('int')
_input0[['CryoSleep', 'VIP']] = _input0[['CryoSleep', 'VIP']].astype('int')
_input1.head()
plt.figure(figsize=(6, 5))
sns.countplot(x='Transported', data=_input1)
x = _input1.loc[_input1.HomePlanet == 'Earth']
y = _input1.loc[_input1.HomePlanet == 'Europa']
z = _input1.loc[_input1.HomePlanet == 'Mars']
a = len(x)
b = len(y)
c = len(z)
y = [a, b, c]
mylabels = ['Earth', 'Europa', 'Mars']
myexplode = [0.1, 0, 0.2]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)
sns.countplot(x=_input1['HomePlanet'], hue=_input1['Transported'])
plt.title('HomePlanet vs Transported')
x = _input1.loc[_input1.HomePlanet == 'Earth']['Transported']
t = sum(x)
y = _input1.loc[_input1.HomePlanet == 'Europa']['Transported']
u = sum(y)
z = _input1.loc[_input1.HomePlanet == 'Mars']['Transported']
v = sum(z)
y = [t, u, v]
mylabels = ['Earth', 'Europa', 'Mars']
myexplode = [0, 0.1, 0.1]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)
print(_input1['Destination'].value_counts())
x = _input1.loc[_input1.Destination == 'TRAPPIST-1e']['Transported']
t = sum(x)
y = _input1.loc[_input1.Destination == '55 Cancri e']['Transported']
u = sum(y)
z = _input1.loc[_input1.Destination == 'PSO J318.5-22']['Transported']
v = sum(z)
y = [t, u, v]
mylabels = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22']
myexplode = [0.1, 0.1, 0.2]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
X = _input1.drop(['PassengerId', 'Transported'], axis=1)
Y = _input1['Transported']
X = pd.get_dummies(X)
_input0.head()
test_x = _input0.drop(['PassengerId'], axis=1)
test_x = pd.get_dummies(test_x)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1)
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)