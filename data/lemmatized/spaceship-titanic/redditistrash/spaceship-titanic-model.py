import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()
_input1['Transported'].value_counts()
_input1['HomePlanet'].value_counts()
sns.countplot(data=_input1, x='HomePlanet', hue='Transported')
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
sns.countplot(data=_input1, x='Destination', hue='Transported')
_input1.isnull().sum()
_input1['Deck'] = _input1['Cabin']
_input1['Side'] = _input1['Cabin']

def AddDeckValues(value):
    if pd.isna(value) == False:
        return value[0]
    else:
        return value

def AddSideValues(value):
    if pd.isna(value):
        return value
    elif value[-1] == 'P':
        return 1
    else:
        return 0
_input1['Deck'] = _input1['Deck'].apply(AddDeckValues)
_input1['Side'] = _input1['Side'].apply(AddSideValues)
_input1 = _input1.drop('Cabin', inplace=False, axis=1)
sns.countplot(data=_input1, x='Side', hue='Transported')
(fig, axes) = plt.subplots(1, 5, figsize=(20, 3))
sns.kdeplot(_input1['RoomService'], ax=axes[0])
sns.kdeplot(_input1['FoodCourt'], ax=axes[1])
sns.kdeplot(_input1['ShoppingMall'], ax=axes[2])
sns.kdeplot(_input1['Spa'], ax=axes[3])
sns.kdeplot(_input1['VRDeck'], ax=axes[4])
fig.tight_layout()
_input1['RoomService'] = _input1['RoomService'].fillna(value=0, inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(value=0, inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(value=0, inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(value=0, inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(value=0, inplace=False)
sns.countplot(x=_input1['VIP'])
_input1['VIP'] = _input1['VIP'].fillna(value=False, inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input1['NumPeople'] = _input1['PassengerId']

def GetNumPeople(ID):
    return int(ID[5:])
_input1['NumPeople'] = _input1['NumPeople'].apply(GetNumPeople)
_input1 = _input1.drop(['Name', 'PassengerId'], inplace=False, axis=1)
_input1['VIP'] = _input1['VIP'].map({True: 1, False: 0})
_input1['CryoSleep'] = _input1['CryoSleep'].map({True: 1, False: 0})
_input1['Transported'] = _input1['Transported'].map({True: 1, False: 0})
(fig, axes) = plt.subplots(1, 5, figsize=(10, 3))
sns.barplot(x='CryoSleep', y='RoomService', data=_input1, ax=axes[0])
sns.barplot(x='CryoSleep', y='FoodCourt', data=_input1, ax=axes[1])
sns.barplot(x='CryoSleep', y='ShoppingMall', data=_input1, ax=axes[2])
sns.barplot(x='CryoSleep', y='Spa', data=_input1, ax=axes[3])
sns.barplot(x='CryoSleep', y='VRDeck', data=_input1, ax=axes[4])
fig.tight_layout()
index = 0

def FillCryoSleepNa(CryoSleep, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck):
    global index
    if pd.isna(CryoSleep[index]) == False:
        ReturnValue = CryoSleep[index]
        index += 1
        return ReturnValue
    for x in [RoomService, FoodCourt, ShoppingMall, Spa, VRDeck]:
        if x[index] != 0:
            index += 1
            return 0.0
    index += 1
    return 1.0
_input1['CryoSleep'] = _input1['CryoSleep'].apply(lambda x: FillCryoSleepNa(_input1.CryoSleep, _input1.RoomService, _input1.FoodCourt, _input1.ShoppingMall, _input1.Spa, _input1.VRDeck))
index = 0
_input1
import random
print('Percentage of unique side values:')
print(_input1['Side'].value_counts(normalize=True))
print('\nPercentage of unique HomePlanet values:')
print(_input1['HomePlanet'].value_counts(normalize=True))
print('\nPercentage of unique Destination values:')
print(_input1['Destination'].value_counts(normalize=True))
print('\nPercentage of unique Deck values:')
print(_input1['Deck'].value_counts(normalize=True))

def FillSideNaValues(value):
    if pd.isna(value):
        return random.choices([0.0, 1.0], weights=(51.6, 48.4))[0]
    return value

def FillHomeNaValues(value):
    if pd.isna(value):
        return random.choices(['Earth', 'Europa', 'Mars'], weights=(54.1, 25.1, 20.8))[0]
    return value

def FillDestinationNaValues(value):
    if pd.isna(value):
        return random.choices(['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'], weights=(69.5, 21.1, 9.4))[0]
    return value

def FillDeckNaValues(value):
    if pd.isna(value):
        return random.choices(['F', 'G', 'E', 'B', 'C', 'D', 'A', 'T'], weights=(32.9, 30.1, 10.3, 9.2, 8.8, 5.6, 3, 0.05))[0]
    return value
_input1['Side'] = _input1['Side'].apply(FillSideNaValues)
_input1['HomePlanet'] = _input1['HomePlanet'].apply(FillHomeNaValues)
_input1['Destination'] = _input1['Destination'].apply(FillDestinationNaValues)
_input1['Deck'] = _input1['Deck'].apply(FillDeckNaValues)
_input1
HomePlanet_Dummies = pd.get_dummies(_input1['HomePlanet'], drop_first=True)
Destination_Dummies = pd.get_dummies(_input1['Destination'], drop_first=True)
Deck_Dummies = pd.get_dummies(_input1['Deck'], drop_first=True)
_input1 = pd.concat([_input1.drop(['HomePlanet', 'Destination', 'Deck'], axis=1), HomePlanet_Dummies, Destination_Dummies, Deck_Dummies], axis=1)
_input1
X = _input1.drop('Transported', axis=1).values
y = _input1['Transported'].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101, test_size=0.3)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(booster='gbtree', gamma=0, learning_rate=0.05, max_depth=4, n_estimators=500, reg_alpha=0, reg_lambda=0)