import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_label = df_train.pop('Transported').astype(int)
print('train: ', df_train.shape[0], ' rows & ', df_train.shape[1], ' columns')
print('test: ', df_test.shape[0], ' rows & ', df_test.shape[1], ' columns')
df_train.head()

def get_values(col):
    return pd.DataFrame({'train': df_train[col].value_counts(), 'test': df_test[col].value_counts()})

def process_data(df):
    df.drop('Name', inplace=True, axis=1)
    df.CryoSleep = df.CryoSleep.astype(int)
    df.HomePlanet = df.HomePlanet.map({'Earth': 0, 'Europa': 1, 'Mars': 2})
    df.Destination = df.Destination.map({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
    df.VIP = df.VIP.astype(int)
    df['Spent'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    age_step = [0, 11, 21, 31, 41, 51, 61, 71, 81, 120]
    age_group = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    df['Age'] = pd.cut(df.Age, age_step, labels=age_group, include_lowest=True)
    return df.pop('PassengerId')

def process_cabin(df):
    try:
        df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    except:
        df[['Deck', 'Num', 'Side']] = ['NaN', 'NaN', 'NaN']
    df.Side = df.Side.map({'P': 0, 'S': 1})
    df.drop('Cabin', inplace=True, axis=1)
    decks = {'F': 0, 'G': 1, 'E': 2, 'B': 3, 'C': 4, 'D': 5, 'A': 6, 'T': 7}
    df['Deck'] = df['Deck'].map(decks)
pd.DataFrame({'train': df_train.isna().sum(), 'test': df_test.isna().sum()})

def fill_by_mean():
    missing_features = list(df_train.columns)
    missing_features.remove('PassengerId')
    missing_features.remove('Name')
    for feature in missing_features:
        if feature == 'Age':
            fill = df_train[feature].mean()
        else:
            fill = df_train[feature].value_counts().index[0]
        df_train[feature] = df_train[feature].fillna(fill)
        df_test[feature] = df_test[feature].fillna(fill)
fill_by_mean()
df_train.info()
get_values('HomePlanet')
get_values('Destination')
id_train = process_data(df_train)
id_test = process_data(df_test)
df_train.head()
process_cabin(df_train)
process_cabin(df_test)
df_train.head()
ct = ColumnTransformer([('normalize', StandardScaler(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Spent'])], remainder='passthrough')