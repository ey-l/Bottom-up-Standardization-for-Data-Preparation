import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import missingno
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.duplicated().sum()
_input1.info()

def update_dataset(df):
    df[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = df['Cabin'].str.split('/', expand=True)
    df['PassengerGroup'] = df['PassengerId'].map(lambda x: x[:4])
    df[['Name_name', 'Name_family']] = df['Name'].str.split(' ', expand=True)
    df = df.drop(['Cabin', 'Name'], axis=1, inplace=False)
    return df
for df in [_input1, _input0]:
    df = update_dataset(df)

def update_by_age(df, column):
    query_str = column + ' > 0'
    min_age = df[['Age', column]].groupby('Age').sum().reset_index().query(query_str).iloc[0, 0]
    df.loc[df['Age'] < min_age, column] = df.loc[df['Age'] < min_age, column].fillna(0)
    return df
for df in [_input1, _input0]:
    for column in ['VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df = update_by_age(df, column)

def update_by_cryo(df, column):
    df.loc[df['CryoSleep'] == True, column] = df.loc[df['CryoSleep'] == True, column].fillna(0)
    return df
for df in [_input1, _input0]:
    for column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df = update_by_age(df, column)
missingno.matrix(_input1, figsize=(15, 5), fontsize=12, color=(0.5, 0.5, 0.5))
_input1.info()

def update_by_mean(df, column):
    if column == 'Age':
        df[column] = df[column].fillna(df[column].median(), inplace=False)
    elif column == 'HomePlanet':
        df[column] = df[column].fillna('Earth', inplace=False)
    elif column == 'Destination':
        df[column] = df[column].fillna('55 Cancri e', inplace=False)
    elif column == 'CryoSleep':
        df[column] = df[column].fillna(False, inplace=False)
    elif column == 'VIP':
        df[column] = df[column].fillna(False, inplace=False)
    else:
        df[column] = df[column].fillna(0, inplace=False)
    return df
for df in [_input1, _input0]:
    for column in ['Age', 'RoomService', 'HomePlanet', 'Destination', 'VIP', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep']:
        df = update_by_mean(df, column)
_input1['Age'] = _input1['Age'].astype('int32')
_input0['Age'] = _input0['Age'].astype('int32')
y_train = _input1['Transported']
features = ['Age', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X_train = pd.get_dummies(_input1[features])
X_test = pd.get_dummies(_input0[features])
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)