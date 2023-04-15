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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Shape of train.csv : ', train_df.shape)
print('Shape of test.csv : ', test_df.shape)
train_df.head()
test_df.head()
train_df.info()
train_df.isnull().sum()
test_df.info()
test_df.isnull().sum()
print(train_df['Age'].median())
print(test_df['Age'].median())
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
column = train_df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    print(column[i].value_counts())
column = train_df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    train_df[i].fillna(train_df[i].mode()[0], inplace=True)
    test_df[i].fillna(test_df[i].mode()[0], inplace=True)
Others = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Others:
    print(train_df[i].median())
Others = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Others:
    train_df[i].fillna(train_df[i].median(), inplace=True)
    test_df[i].fillna(test_df[i].median(), inplace=True)
train_df = train_df.drop(columns=['Cabin', 'Name'], axis=1)
test_df = test_df.drop(columns=['Cabin', 'Name'], axis=1)
print('Shape of train.csv : ', train_df.shape)
print('Shape of test.csv : ', test_df.shape)
train_df[['CryoSleep', 'VIP']] = train_df[['CryoSleep', 'VIP']].astype('int')
test_df[['CryoSleep', 'VIP']] = test_df[['CryoSleep', 'VIP']].astype('int')
train_df.head()
plt.figure(figsize=(6, 5))
sns.countplot(x='Transported', data=train_df)
x = train_df.loc[train_df.HomePlanet == 'Earth']
y = train_df.loc[train_df.HomePlanet == 'Europa']
z = train_df.loc[train_df.HomePlanet == 'Mars']
a = len(x)
b = len(y)
c = len(z)
y = [a, b, c]
mylabels = ['Earth', 'Europa', 'Mars']
myexplode = [0.1, 0, 0.2]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)

sns.countplot(x=train_df['HomePlanet'], hue=train_df['Transported'])
plt.title('HomePlanet vs Transported')

x = train_df.loc[train_df.HomePlanet == 'Earth']['Transported']
t = sum(x)
y = train_df.loc[train_df.HomePlanet == 'Europa']['Transported']
u = sum(y)
z = train_df.loc[train_df.HomePlanet == 'Mars']['Transported']
v = sum(z)
y = [t, u, v]
mylabels = ['Earth', 'Europa', 'Mars']
myexplode = [0, 0.1, 0.1]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)

print(train_df['Destination'].value_counts())
x = train_df.loc[train_df.Destination == 'TRAPPIST-1e']['Transported']
t = sum(x)
y = train_df.loc[train_df.Destination == '55 Cancri e']['Transported']
u = sum(y)
z = train_df.loc[train_df.Destination == 'PSO J318.5-22']['Transported']
v = sum(z)
y = [t, u, v]
mylabels = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22']
myexplode = [0.1, 0.1, 0.2]
plt.pie(y, labels=mylabels, explode=myexplode, shadow=True)

sns.countplot(data=train_df, x='CryoSleep', hue='Transported')
X = train_df.drop(['PassengerId', 'Transported'], axis=1)
Y = train_df['Transported']
X = pd.get_dummies(X)
test_df.head()
test_x = test_df.drop(['PassengerId'], axis=1)
test_x = pd.get_dummies(test_x)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1)
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)