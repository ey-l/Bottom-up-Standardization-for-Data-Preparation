import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train shape: ', _input1.shape)
print('Test shape: ', _input0.shape)
_input1.head(10)
_input1.info()
_input1['HomePlanet'].value_counts()
_input1['Destination'].value_counts()
sns.countplot(_input1['Transported'])
sns.distplot(_input1['Age'], bins=25)
sns.countplot(_input1['VIP'])
sns.displot(data=_input1, x='Age', bins=25, hue='VIP', palette='viridis')
sns.boxplot(data=_input1[_input1['RoomService'] != 0.0], y='RoomService')
sns.boxplot(data=_input1[_input1['FoodCourt'] != 0.0], y='FoodCourt')
sns.boxplot(data=_input1[_input1['ShoppingMall'] != 0.0], y='ShoppingMall')
sns.boxplot(data=_input1[_input1['Spa'] != 0.0], y='Spa')
_input1.head(10)
_input1['Route'] = _input1.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
_input1['Cabin_deck'] = _input1['Cabin'].str[0]
_input1['Cabin_num'] = _input1['Cabin'].str[2]
_input1['Cabin_side'] = _input1['Cabin'].str[-1]
_input0['Route'] = _input0.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
_input0['Cabin_deck'] = _input0['Cabin'].str[0]
_input0['Cabin_num'] = _input0['Cabin'].str[2]
_input0['Cabin_side'] = _input0['Cabin'].str[-1]
_input1.isnull().sum()
_input1.groupby('Route').count()

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
_input1['Destination'] = _input1.apply(lambda x: fillMissDes(x['Route'], x['Destination']), axis=1)
_input1['HomePlanet'] = _input1.apply(lambda x: fillMissHome(x['Route'], x['HomePlanet']), axis=1)
_input1['Route'] = _input1.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
_input0['Destination'] = _input0.apply(lambda x: fillMissDes(x['Route'], x['Destination']), axis=1)
_input0['HomePlanet'] = _input0.apply(lambda x: fillMissHome(x['Route'], x['HomePlanet']), axis=1)
_input0['Route'] = _input0.apply(lambda x: '%s to %s' % (x['HomePlanet'], x['Destination']), axis=1)
_input1.isnull().sum()
plt.figure(figsize=(10, 6))
sns.countplot(data=_input1, x='Transported', hue='Route')
sns.countplot(data=_input1, x='CryoSleep', hue='Route')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(value=False, inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(value=False, inplace=False)
sns.distplot(_input1['Age'], bins=30)
_input1['Age'].mean()
pd.DataFrame(_input1.groupby(['Cabin_deck', 'Cabin_num'])['PassengerId'].count()).unstack('Cabin_num')
sns.boxplot(data=_input1, x='VIP', y='VRDeck')
sns.countplot(data=_input1, x='Transported', hue='Cabin_side')
sns.countplot(data=_input1, x='Transported', hue='Cabin_deck')
sns.countplot(data=_input1, x='Transported', hue='Cabin_num')
_input1['Age'] = _input1['Age'].fillna(value=29, inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(value=False, inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(value=0, inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(value=0, inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(value=0, inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(value=0, inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(value=0, inplace=False)
_input1 = _input1.drop('Name', axis=1, inplace=False)
df = _input1.drop(['Cabin', 'Cabin_deck', 'Cabin_side', 'Cabin_num'], axis=1)
_input0['Age'] = _input0['Age'].fillna(value=29, inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(value=False, inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(value=0, inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(value=0, inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(value=0, inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(value=0, inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(value=0, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop(['Cabin', 'Cabin_deck', 'Cabin_side', 'Cabin_num'], axis=1, inplace=False)
df.isnull().sum()
TrueFalse_dict = {True: 1, False: 0}
df['CryoSleep'] = df['CryoSleep'].map(TrueFalse_dict)
df['VIP'] = df['VIP'].map(TrueFalse_dict)
df['Transported'] = df['Transported'].map(TrueFalse_dict)
_input0['CryoSleep'] = _input0['CryoSleep'].map(TrueFalse_dict)
_input0['VIP'] = _input0['VIP'].map(TrueFalse_dict)
df.dtypes
df = df.drop(['PassengerId', 'HomePlanet', 'Destination'], axis=1, inplace=False)
_input0 = _input0.drop(['HomePlanet', 'Destination'], axis=1, inplace=False)
_input0.columns
Route_dummy = pd.get_dummies(df['Route'], drop_first=True)
df = pd.concat([df, Route_dummy], axis=1).drop('Route', axis=1)
_input0 = pd.concat([_input0, Route_dummy], axis=1).drop('Route', axis=1)
from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns != 'Transported']
y = df['Transported']
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.3, random_state=123)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()