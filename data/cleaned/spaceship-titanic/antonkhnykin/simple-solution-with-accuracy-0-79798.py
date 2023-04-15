import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import missingno
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.duplicated().sum()
train_data.info()


def update_dataset(df):
    df[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = df['Cabin'].str.split('/', expand=True)
    df['PassengerGroup'] = df['PassengerId'].map(lambda x: x[:4])
    df[['Name_name', 'Name_family']] = df['Name'].str.split(' ', expand=True)
    df.drop(['Cabin', 'Name'], axis=1, inplace=True)
    return df
for df in [train_data, test_data]:
    df = update_dataset(df)

def update_by_age(df, column):
    query_str = column + ' > 0'
    min_age = df[['Age', column]].groupby('Age').sum().reset_index().query(query_str).iloc[0, 0]
    df.loc[df['Age'] < min_age, column] = df.loc[df['Age'] < min_age, column].fillna(0)
    return df
for df in [train_data, test_data]:
    for column in ['VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df = update_by_age(df, column)

def update_by_cryo(df, column):
    df.loc[df['CryoSleep'] == True, column] = df.loc[df['CryoSleep'] == True, column].fillna(0)
    return df
for df in [train_data, test_data]:
    for column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df = update_by_age(df, column)
missingno.matrix(train_data, figsize=(15, 5), fontsize=12, color=(0.5, 0.5, 0.5))

train_data.info()

def update_by_mean(df, column):
    if column == 'Age':
        df[column].fillna(df[column].median(), inplace=True)
    elif column == 'HomePlanet':
        df[column].fillna('Earth', inplace=True)
    elif column == 'Destination':
        df[column].fillna('55 Cancri e', inplace=True)
    elif column == 'CryoSleep':
        df[column].fillna(False, inplace=True)
    elif column == 'VIP':
        df[column].fillna(False, inplace=True)
    else:
        df[column].fillna(0, inplace=True)
    return df
for df in [train_data, test_data]:
    for column in ['Age', 'RoomService', 'HomePlanet', 'Destination', 'VIP', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CryoSleep']:
        df = update_by_mean(df, column)
train_data['Age'] = train_data['Age'].astype('int32')
test_data['Age'] = test_data['Age'].astype('int32')
y_train = train_data['Transported']
features = ['Age', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)