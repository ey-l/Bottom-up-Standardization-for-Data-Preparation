import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train
df_train.info()
df_train['Transported'].value_counts()
df_train['HomePlanet'].value_counts()
sns.countplot(data=df_train, x='HomePlanet', hue='Transported')
sns.countplot(data=df_train, x='CryoSleep', hue='Transported')
sns.countplot(data=df_train, x='Destination', hue='Transported')
df_train.isnull().sum()
df_train['Deck'] = df_train['Cabin']
df_train['Side'] = df_train['Cabin']

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
df_train['Deck'] = df_train['Deck'].apply(AddDeckValues)
df_train['Side'] = df_train['Side'].apply(AddSideValues)
df_train.drop('Cabin', inplace=True, axis=1)
sns.countplot(data=df_train, x='Side', hue='Transported')
(fig, axes) = plt.subplots(1, 5, figsize=(20, 3))
sns.kdeplot(df_train['RoomService'], ax=axes[0])
sns.kdeplot(df_train['FoodCourt'], ax=axes[1])
sns.kdeplot(df_train['ShoppingMall'], ax=axes[2])
sns.kdeplot(df_train['Spa'], ax=axes[3])
sns.kdeplot(df_train['VRDeck'], ax=axes[4])
fig.tight_layout()
df_train['RoomService'].fillna(value=0, inplace=True)
df_train['FoodCourt'].fillna(value=0, inplace=True)
df_train['ShoppingMall'].fillna(value=0, inplace=True)
df_train['Spa'].fillna(value=0, inplace=True)
df_train['VRDeck'].fillna(value=0, inplace=True)
sns.countplot(x=df_train['VIP'])
df_train['VIP'].fillna(value=False, inplace=True)
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['NumPeople'] = df_train['PassengerId']

def GetNumPeople(ID):
    return int(ID[5:])
df_train['NumPeople'] = df_train['NumPeople'].apply(GetNumPeople)
df_train.drop(['Name', 'PassengerId'], inplace=True, axis=1)
df_train['VIP'] = df_train['VIP'].map({True: 1, False: 0})
df_train['CryoSleep'] = df_train['CryoSleep'].map({True: 1, False: 0})
df_train['Transported'] = df_train['Transported'].map({True: 1, False: 0})
(fig, axes) = plt.subplots(1, 5, figsize=(10, 3))
sns.barplot(x='CryoSleep', y='RoomService', data=df_train, ax=axes[0])
sns.barplot(x='CryoSleep', y='FoodCourt', data=df_train, ax=axes[1])
sns.barplot(x='CryoSleep', y='ShoppingMall', data=df_train, ax=axes[2])
sns.barplot(x='CryoSleep', y='Spa', data=df_train, ax=axes[3])
sns.barplot(x='CryoSleep', y='VRDeck', data=df_train, ax=axes[4])
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
df_train['CryoSleep'] = df_train['CryoSleep'].apply(lambda x: FillCryoSleepNa(df_train.CryoSleep, df_train.RoomService, df_train.FoodCourt, df_train.ShoppingMall, df_train.Spa, df_train.VRDeck))
index = 0
df_train
import random
print('Percentage of unique side values:')
print(df_train['Side'].value_counts(normalize=True))
print('\nPercentage of unique HomePlanet values:')
print(df_train['HomePlanet'].value_counts(normalize=True))
print('\nPercentage of unique Destination values:')
print(df_train['Destination'].value_counts(normalize=True))
print('\nPercentage of unique Deck values:')
print(df_train['Deck'].value_counts(normalize=True))

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
df_train['Side'] = df_train['Side'].apply(FillSideNaValues)
df_train['HomePlanet'] = df_train['HomePlanet'].apply(FillHomeNaValues)
df_train['Destination'] = df_train['Destination'].apply(FillDestinationNaValues)
df_train['Deck'] = df_train['Deck'].apply(FillDeckNaValues)
df_train
HomePlanet_Dummies = pd.get_dummies(df_train['HomePlanet'], drop_first=True)
Destination_Dummies = pd.get_dummies(df_train['Destination'], drop_first=True)
Deck_Dummies = pd.get_dummies(df_train['Deck'], drop_first=True)
df_train = pd.concat([df_train.drop(['HomePlanet', 'Destination', 'Deck'], axis=1), HomePlanet_Dummies, Destination_Dummies, Deck_Dummies], axis=1)
df_train
X = df_train.drop('Transported', axis=1).values
y = df_train['Transported'].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101, test_size=0.3)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(booster='gbtree', gamma=0, learning_rate=0.05, max_depth=4, n_estimators=500, reg_alpha=0, reg_lambda=0)