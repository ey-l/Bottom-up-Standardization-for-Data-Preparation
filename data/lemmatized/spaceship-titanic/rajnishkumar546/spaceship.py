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
_input0.head()
_input1.shape
_input0.shape
_input1.info()
_input0.info()
_input1.isnull().sum()
_input0.isnull().sum()
plt.figure(figsize=(10, 6))
sns.heatmap(_input1.corr(), annot=True)
_input1['Transported'].value_counts()
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1, inplace=False)
train2 = _input1.copy()
train2.head()
train2 = train2.drop('CryoSleep', axis=1, inplace=False)
train2 = train2.drop(['HomePlanet', 'Cabin', 'Destination', 'VIP', 'Transported'], axis=1, inplace=False)
sns.pairplot(train2)
_input1['HomePlanet'].value_counts()
sns.displot(x=_input1['HomePlanet'], data=_input1, kde=True)
_input1['CryoSleep'].value_counts()
_input1['Cabin'].value_counts()
_input1['Destination'].value_counts()
sns.displot(x=_input1['Destination'], data=_input1, kde=True)
_input1['VIP'].value_counts()

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
_input1['Transported'] = _input1['Transported'].apply(transport)
_input1
_input1.shape
_input0.shape
sns.histplot(x=_input1['HomePlanet'], y=_input1['Transported'], data=_input1)
concat = pd.concat([_input1, _input0], axis=0)
concat.head()
concat.tail()
concat.shape
_input1[_input1['HomePlanet'] == 'Earth']['Transported'].value_counts()
concat.isnull().sum()
concat['HomePlanet'].mode()
concat['HomePlanet'] = concat['HomePlanet'].fillna(concat['HomePlanet'].mode()[0])
concat = concat.drop('ShoppingMall', axis=1, inplace=False)
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
_input1['Destination'].mode()
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
test_set = test_set.drop('Transported', axis=1, inplace=False)
X = train_set.drop('Transported', axis=1)
y = train_set['Transported']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)