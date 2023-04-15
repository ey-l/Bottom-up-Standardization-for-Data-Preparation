import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (12, 6)
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')




fig = sns.countplot(data=train, x='CryoSleep', hue='Transported')
fig = sns.countplot(data=train, x='HomePlanet', hue='Transported')
fig = sns.countplot(data=train, x='Destination', hue='Transported')

train['Deck'] = train['Cabin']
train['Side'] = train['Cabin']

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
train['Deck'] = train['Deck'].apply(AddDeckValues)
train['Side'] = train['Side'].apply(AddSideValues)
train.drop('Cabin', inplace=True, axis=1)
fig = sns.countplot(data=train, x='Side', hue='Transported')
(fig, axes) = plt.subplots(1, 5, figsize=(20, 3))
sns.kdeplot(train['RoomService'], ax=axes[0])
sns.kdeplot(train['FoodCourt'], ax=axes[1])
sns.kdeplot(train['ShoppingMall'], ax=axes[2])
sns.kdeplot(train['Spa'], ax=axes[3])
sns.kdeplot(train['VRDeck'], ax=axes[4])
fig.tight_layout()
train['RoomService'].fillna(value=0, inplace=True)
train['FoodCourt'].fillna(value=0, inplace=True)
train['ShoppingMall'].fillna(value=0, inplace=True)
train['Spa'].fillna(value=0, inplace=True)
train['VRDeck'].fillna(value=0, inplace=True)
fig = sns.countplot(x=train['VIP'])
train['VIP'].fillna(value=False, inplace=True)
train['Age'] = train['Age'].fillna(train['Age'].median())
train['NumPeople'] = train['PassengerId']

def GetNumPeople(ID):
    return int(ID[5:])
train['NumPeople'] = train['NumPeople'].apply(GetNumPeople)
train.drop(['Name', 'PassengerId'], inplace=True, axis=1)
train['VIP'] = train['VIP'].map({True: 1, False: 0})
train['CryoSleep'] = train['CryoSleep'].map({True: 1, False: 0})
train['Transported'] = train['Transported'].map({True: 1, False: 0})
(fig, axes) = plt.subplots(1, 5, figsize=(10, 3))
sns.barplot(x='CryoSleep', y='RoomService', data=train, ax=axes[0])
sns.barplot(x='CryoSleep', y='FoodCourt', data=train, ax=axes[1])
sns.barplot(x='CryoSleep', y='ShoppingMall', data=train, ax=axes[2])
sns.barplot(x='CryoSleep', y='Spa', data=train, ax=axes[3])
sns.barplot(x='CryoSleep', y='VRDeck', data=train, ax=axes[4])
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
train['CryoSleep'] = train['CryoSleep'].apply(lambda x: FillCryoSleepNa(train.CryoSleep, train.RoomService, train.FoodCourt, train.ShoppingMall, train.Spa, train.VRDeck))
index = 0

import random
print('Percentage of unique side values:')
print(train['Side'].value_counts(normalize=True))
print('\nPercentage of unique HomePlanet values:')
print(train['HomePlanet'].value_counts(normalize=True))
print('\nPercentage of unique Destination values:')
print(train['Destination'].value_counts(normalize=True))
print('\nPercentage of unique Deck values:')
print(train['Deck'].value_counts(normalize=True))

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
train['Side'] = train['Side'].apply(FillSideNaValues)
train['HomePlanet'] = train['HomePlanet'].apply(FillHomeNaValues)
train['Destination'] = train['Destination'].apply(FillDestinationNaValues)
train['Deck'] = train['Deck'].apply(FillDeckNaValues)

HomePlanet_Dummies = pd.get_dummies(train['HomePlanet'], drop_first=True)
Destination_Dummies = pd.get_dummies(train['Destination'], drop_first=True)
Deck_Dummies = pd.get_dummies(train['Deck'], drop_first=True)
train = pd.concat([train.drop(['HomePlanet', 'Destination', 'Deck'], axis=1), HomePlanet_Dummies, Destination_Dummies, Deck_Dummies], axis=1)

X = train.drop('Transported', axis=1).values
y = train['Transported'].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101, test_size=0.3)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(booster='gbtree', gamma=0, learning_rate=0.05, max_depth=4, n_estimators=500, reg_alpha=0, reg_lambda=0)