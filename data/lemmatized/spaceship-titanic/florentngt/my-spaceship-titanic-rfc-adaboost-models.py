import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_label = _input1.pop('Transported').astype(int)
print('train: ', _input1.shape[0], ' rows & ', _input1.shape[1], ' columns')
print('test: ', _input0.shape[0], ' rows & ', _input0.shape[1], ' columns')
_input1.head()

def get_values(col):
    return pd.DataFrame({'train': _input1[col].value_counts(), 'test': _input0[col].value_counts()})

def process_data(df):
    df = df.drop('Name', inplace=False, axis=1)
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
    df = df.drop('Cabin', inplace=False, axis=1)
    decks = {'F': 0, 'G': 1, 'E': 2, 'B': 3, 'C': 4, 'D': 5, 'A': 6, 'T': 7}
    df['Deck'] = df['Deck'].map(decks)
pd.DataFrame({'train': _input1.isna().sum(), 'test': _input0.isna().sum()})

def fill_by_mean():
    missing_features = list(_input1.columns)
    missing_features.remove('PassengerId')
    missing_features.remove('Name')
    for feature in missing_features:
        if feature == 'Age':
            fill = _input1[feature].mean()
        else:
            fill = _input1[feature].value_counts().index[0]
        _input1[feature] = _input1[feature].fillna(fill)
        _input0[feature] = _input0[feature].fillna(fill)
fill_by_mean()
_input1.info()
get_values('HomePlanet')
get_values('Destination')
id_train = process_data(_input1)
id_test = process_data(_input0)
_input1.head()
process_cabin(_input1)
process_cabin(_input0)
_input1.head()
ct = ColumnTransformer([('normalize', StandardScaler(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Spent'])], remainder='passthrough')