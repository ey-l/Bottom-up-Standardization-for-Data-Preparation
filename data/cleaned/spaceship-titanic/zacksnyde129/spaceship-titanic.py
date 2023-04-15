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
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test
train
train.info()
train.describe()
sns.countplot(x='HomePlanet', data=train)
sns.countplot(x='CryoSleep', data=train)
sns.countplot(x='Destination', data=train)
sns.countplot(x='VIP', data=train)
sns.countplot(x='Transported', data=train)
sns.countplot(x='Transported', hue='CryoSleep', data=train)
sns.countplot(x='Transported', hue='Destination', data=train)
sns.countplot(x='Transported', hue='HomePlanet', data=train)
sns.countplot(x='Transported', hue='VIP', data=train)
np.array(train.isnull().sum())
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train_drop = train.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
train_drop.head()
test = test.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
test.head()
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
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['RoomService'] = test['RoomService'].fillna(test['RoomService'].mean())
test['FoodCourt'] = test['FoodCourt'].fillna(test['FoodCourt'].mean())
test['ShoppingMall'] = test['ShoppingMall'].fillna(test['ShoppingMall'].mean())
test['Spa'] = test['Spa'].fillna(test['Spa'].mean())
test['VRDeck'] = test['VRDeck'].fillna(test['VRDeck'].mean())
test['HomePlanet'] = test['HomePlanet'].fillna(test['HomePlanet'].mode()[0])
test['CryoSleep'] = test['CryoSleep'].fillna(test['CryoSleep'].mode()[0])
test['Destination'] = test['Destination'].fillna(test['Destination'].mode()[0])
test['VIP'] = test['VIP'].fillna(test['VIP'].mode()[0])
sns.heatmap(train_drop.isnull(), yticklabels=False, cbar=False)
train_drop.HomePlanet.replace('Earth', 0, inplace=True)
train_drop.HomePlanet.replace('Europa', 1, inplace=True)
train_drop.HomePlanet.replace('Mars', 2, inplace=True)
train_drop['CryoSleep'] = train_drop['CryoSleep'].map(int)
train_drop['VIP'] = train_drop['VIP'].map(int)
train_drop.Transported = train_drop.Transported.replace({True: 0, False: 1})
train_drop.Destination.replace('TRAPPIST-1e', 0, inplace=True)
train_drop.Destination.replace('55 Cancri e', 1, inplace=True)
train_drop.Destination.replace('PSO J318.5-22', 2, inplace=True)
train_drop
test.HomePlanet.replace('Earth', 0, inplace=True)
test.HomePlanet.replace('Europa', 1, inplace=True)
test.HomePlanet.replace('Mars', 2, inplace=True)
test['CryoSleep'] = test['CryoSleep'].map(int)
test['VIP'] = test['VIP'].map(int)
test.Destination.replace('TRAPPIST-1e', 0, inplace=True)
test.Destination.replace('55 Cancri e', 1, inplace=True)
test.Destination.replace('PSO J318.5-22', 2, inplace=True)
test.head()
Y_train = train['Transported']
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X_train = train_drop[features]
X_test = test[features]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)