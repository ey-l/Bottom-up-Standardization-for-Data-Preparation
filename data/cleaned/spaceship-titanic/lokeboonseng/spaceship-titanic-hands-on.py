import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
train.head(10)
train.info()
train['HomePlanet'].value_counts()
train['Destination'].value_counts()
sns.countplot(train['Transported'])
sns.distplot(train['Age'], bins=25)
sns.countplot(train['VIP'])
sns.displot(data=train, x='Age', bins=25, hue='VIP', palette='viridis')
sns.boxplot(data=train[train['RoomService'] != 0.0], y='RoomService')
sns.boxplot(data=train[train['FoodCourt'] != 0.0], y='FoodCourt')
sns.boxplot(data=train[train['ShoppingMall'] != 0.0], y='ShoppingMall')
sns.boxplot(data=train[train['Spa'] != 0.0], y='Spa')
train.head(10)
train['Route'] = train.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
train['Cabin_deck'] = train['Cabin'].str[0]
train['Cabin_num'] = train['Cabin'].str[2]
train['Cabin_side'] = train['Cabin'].str[-1]
test['Route'] = test.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
test['Cabin_deck'] = test['Cabin'].str[0]
test['Cabin_num'] = test['Cabin'].str[2]
test['Cabin_side'] = test['Cabin'].str[-1]
train.isnull().sum()
train.groupby('Route').count()

def fillMissDes(route, Des):
    if route == 'Earth to nan':
        return 'TRAPPIST-1e'
    elif route == 'Europa to nan':
        return 'TRAPPIST-1e'
    elif route == 'Mars to nan':
        return 'TRAPPIST-1e'
    elif route == 'nan to nan':
        return 'TRAPPIST-1e'
    else:
        return Des

def fillMissHome(route, Home):
    if route == 'nan to 55 Cancri e':
        return 'Europa'
    elif route == 'nan to PSO J318.5-22':
        return 'Earth'
    elif route == 'nan to TRAPPIST-1e':
        return 'Earth'
    elif route == 'nan to nan':
        return 'Earth'
    else:
        return Home
train['Destination'] = train.apply(lambda x: fillMissDes(x['Route'], x['Destination']), axis=1)
train['HomePlanet'] = train.apply(lambda x: fillMissHome(x['Route'], x['HomePlanet']), axis=1)
train['Route'] = train.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
test['Destination'] = test.apply(lambda x: fillMissDes(x['Route'], x['Destination']), axis=1)
test['HomePlanet'] = test.apply(lambda x: fillMissHome(x['Route'], x['HomePlanet']), axis=1)
test['Route'] = test.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
train.isnull().sum()
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Transported', hue='Route')
sns.countplot(data=train, x='CryoSleep', hue='Route')
train['CryoSleep'].fillna(value=False, inplace=True)
test['CryoSleep'].fillna(value=False, inplace=True)
sns.distplot(train['Age'], bins=30)
train['Age'].mean()
pd.DataFrame(train.groupby(['Cabin_deck', 'Cabin_num'])['PassengerId'].count()).unstack('Cabin_num')
sns.boxplot(data=train, x='VIP', y='VRDeck')
sns.countplot(data=train, x='Transported', hue='Cabin_side')
sns.countplot(data=train, x='Transported', hue='Cabin_deck')
sns.countplot(data=train, x='Transported', hue='Cabin_num')
train['Age'].fillna(value=29, inplace=True)
train['VIP'].fillna(value=False, inplace=True)
train['RoomService'].fillna(value=0, inplace=True)
train['FoodCourt'].fillna(value=0, inplace=True)
train['Spa'].fillna(value=0, inplace=True)
train['ShoppingMall'].fillna(value=0, inplace=True)
train['VRDeck'].fillna(value=0, inplace=True)
train.drop('Name', axis=1, inplace=True)
df = train.drop(['Cabin', 'Cabin_deck', 'Cabin_side', 'Cabin_num'], axis=1)
test['Age'].fillna(value=29, inplace=True)
test['VIP'].fillna(value=False, inplace=True)
test['RoomService'].fillna(value=0, inplace=True)
test['FoodCourt'].fillna(value=0, inplace=True)
test['Spa'].fillna(value=0, inplace=True)
test['ShoppingMall'].fillna(value=0, inplace=True)
test['VRDeck'].fillna(value=0, inplace=True)
test.drop('Name', axis=1, inplace=True)
test.drop(['Cabin', 'Cabin_deck', 'Cabin_side', 'Cabin_num'], axis=1, inplace=True)
df.isnull().sum()
TrueFalse_dict = {True: 1, False: 0}
df['CryoSleep'] = df['CryoSleep'].map(TrueFalse_dict)
df['VIP'] = df['VIP'].map(TrueFalse_dict)
df['Transported'] = df['Transported'].map(TrueFalse_dict)
test['CryoSleep'] = test['CryoSleep'].map(TrueFalse_dict)
test['VIP'] = test['VIP'].map(TrueFalse_dict)
df.dtypes
df.drop(['PassengerId', 'HomePlanet', 'Destination'], axis=1, inplace=True)
test.drop(['HomePlanet', 'Destination'], axis=1, inplace=True)
test.columns
Route_dummy = pd.get_dummies(df['Route'], drop_first=True)
df = pd.concat([df, Route_dummy], axis=1).drop('Route', axis=1)
test = pd.concat([test, Route_dummy], axis=1).drop('Route', axis=1)
from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns != 'Transported']
y = df['Transported']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.3, random_state=123)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()