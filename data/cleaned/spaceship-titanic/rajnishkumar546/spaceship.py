import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submit = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True)
train['Transported'].value_counts()
train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name'], axis=1, inplace=True)
train2 = train.copy()
train2.head()
train2.drop('CryoSleep', axis=1, inplace=True)
train2.drop(['HomePlanet', 'Cabin', 'Destination', 'VIP', 'Transported'], axis=1, inplace=True)
sns.pairplot(train2)
train['HomePlanet'].value_counts()
sns.displot(x=train['HomePlanet'], data=train, kde=True)
train['CryoSleep'].value_counts()
train['Cabin'].value_counts()
train['Destination'].value_counts()
sns.displot(x=train['Destination'], data=train, kde=True)
train['VIP'].value_counts()

def transport(x):
    if x == True:
        return 1
    if x == False:
        return 0

def transports(x):
    if x == 'True':
        return 1
    if x == 'False':
        return 0
train['Transported'] = train['Transported'].apply(transport)
train
train.shape
test.shape
sns.histplot(x=train['HomePlanet'], y=train['Transported'], data=train)
concat = pd.concat([train, test], axis=0)
concat.head()
concat.tail()
concat.shape
train[train['HomePlanet'] == 'Earth']['Transported'].value_counts()
concat.isnull().sum()
concat['HomePlanet'].mode()
concat['HomePlanet'] = concat['HomePlanet'].fillna(concat['HomePlanet'].mode()[0])
concat.drop('ShoppingMall', axis=1, inplace=True)
concat['CryoSleep'].value_counts()
concat['CryoSleep'].mode()

def cryo(x):
    if x == 'True':
        return 1
    else:
        return 0
concat['CryoSleep'] = concat['CryoSleep'].fillna(concat['CryoSleep'].mode()[0])
concat['CryoSleep'] = concat['CryoSleep'].apply(cryo)
concat.head()
train['Destination'].mode()
concat['Destination'] = concat['Destination'].fillna(concat['Destination'].mode()[0])
concat['Age'] = concat['Age'].fillna(concat['Age'].mean())
concat['Cabin'] = concat['Cabin'].fillna(concat['Cabin'].mode()[0])
concat['VIP']
concat['VIP'] = concat['VIP'].fillna(concat['VIP'].mode()[0])
concat['VIP'] = concat['VIP'].apply(cryo)
concat.isnull().sum()
concat['RoomService'] = concat['RoomService'].fillna(concat['RoomService'].mean())
concat['FoodCourt'] = concat['FoodCourt'].fillna(concat['FoodCourt'].mean())
concat['Spa'] = concat['Spa'].fillna(concat['Spa'].mean())
concat['VRDeck'] = concat['VRDeck'].fillna(concat['VRDeck'].mean())
concat.isnull().sum()
concat.info()

def planet(x):
    if x == 'Earth':
        return 1
    elif x == 'Europa':
        return 2
    else:
        return 3
concat['HomePlanet'] = concat['HomePlanet'].apply(planet)
concat['Destination'].value_counts()

def dest(x):
    if x == 'TRAPPIST-1e':
        return 1
    elif x == '55 Cancri e':
        return 2
    else:
        return 3
concat['Destination'] = concat['Destination'].apply(dest)

def cabin(x):
    b = str(x)
    c = b.split('/')[0]
    return c
concat['Cabin'] = concat['Cabin'].apply(cabin)
concat['Cabin'].value_counts()

def cabin2(x):
    if x == 'F':
        return 1
    elif x == 'G':
        return 2
    elif x == 'E':
        return 3
    elif x == 'B':
        return 4
    elif x == 'C':
        return 5
    elif x == 'D':
        return 6
    elif x == 'A':
        return 7
    else:
        return 8
concat['Cabin'] = concat['Cabin'].apply(cabin2)
concat.info()
train_set = concat.iloc[0:8693, :]
train_set.shape
test_set = concat.iloc[8693:12970, :]
test_set.shape
test_set.drop('Transported', axis=1, inplace=True)
X = train_set.drop('Transported', axis=1)
y = train_set['Transported']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)