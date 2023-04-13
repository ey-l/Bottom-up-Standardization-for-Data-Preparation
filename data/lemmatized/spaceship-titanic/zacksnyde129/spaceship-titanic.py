import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0
_input1
_input1.info()
_input1.describe()
sns.countplot(x='HomePlanet', data=_input1)
sns.countplot(x='CryoSleep', data=_input1)
sns.countplot(x='Destination', data=_input1)
sns.countplot(x='VIP', data=_input1)
sns.countplot(x='Transported', data=_input1)
sns.countplot(x='Transported', hue='CryoSleep', data=_input1)
sns.countplot(x='Transported', hue='Destination', data=_input1)
sns.countplot(x='Transported', hue='HomePlanet', data=_input1)
sns.countplot(x='Transported', hue='VIP', data=_input1)
np.array(_input1.isnull().sum())
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
train_drop = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
train_drop.head()
_input0 = _input0.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
_input0.head()
train_drop['Age'] = train_drop['Age'].fillna(train_drop['Age'].mean())
train_drop['RoomService'] = train_drop['RoomService'].fillna(train_drop['RoomService'].mean())
train_drop['FoodCourt'] = train_drop['FoodCourt'].fillna(train_drop['FoodCourt'].mean())
train_drop['ShoppingMall'] = train_drop['ShoppingMall'].fillna(train_drop['ShoppingMall'].mean())
train_drop['Spa'] = train_drop['Spa'].fillna(train_drop['Spa'].mean())
train_drop['VRDeck'] = train_drop['VRDeck'].fillna(train_drop['VRDeck'].mean())
sns.heatmap(train_drop.isnull(), yticklabels=False, cbar=False)
train_drop['HomePlanet'] = train_drop['HomePlanet'].fillna(train_drop['HomePlanet'].mode()[0])
train_drop['CryoSleep'] = train_drop['CryoSleep'].fillna(train_drop['CryoSleep'].mode()[0])
train_drop['Destination'] = train_drop['Destination'].fillna(train_drop['Destination'].mode()[0])
train_drop['VIP'] = train_drop['VIP'].fillna(train_drop['VIP'].mode()[0])
sns.heatmap(train_drop.isnull(), yticklabels=False, cbar=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean())
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean())
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean())
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean())
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean())
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0])
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0])
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0])
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0])
sns.heatmap(train_drop.isnull(), yticklabels=False, cbar=False)
train_drop.HomePlanet = train_drop.HomePlanet.replace('Earth', 0, inplace=False)
train_drop.HomePlanet = train_drop.HomePlanet.replace('Europa', 1, inplace=False)
train_drop.HomePlanet = train_drop.HomePlanet.replace('Mars', 2, inplace=False)
train_drop['CryoSleep'] = train_drop['CryoSleep'].map(int)
train_drop['VIP'] = train_drop['VIP'].map(int)
train_drop.Transported = train_drop.Transported.replace({True: 0, False: 1})
train_drop.Destination = train_drop.Destination.replace('TRAPPIST-1e', 0, inplace=False)
train_drop.Destination = train_drop.Destination.replace('55 Cancri e', 1, inplace=False)
train_drop.Destination = train_drop.Destination.replace('PSO J318.5-22', 2, inplace=False)
train_drop
_input0.HomePlanet = _input0.HomePlanet.replace('Earth', 0, inplace=False)
_input0.HomePlanet = _input0.HomePlanet.replace('Europa', 1, inplace=False)
_input0.HomePlanet = _input0.HomePlanet.replace('Mars', 2, inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].map(int)
_input0['VIP'] = _input0['VIP'].map(int)
_input0.Destination = _input0.Destination.replace('TRAPPIST-1e', 0, inplace=False)
_input0.Destination = _input0.Destination.replace('55 Cancri e', 1, inplace=False)
_input0.Destination = _input0.Destination.replace('PSO J318.5-22', 2, inplace=False)
_input0.head()
Y_train = _input1['Transported']
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X_train = train_drop[features]
X_test = _input0[features]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)