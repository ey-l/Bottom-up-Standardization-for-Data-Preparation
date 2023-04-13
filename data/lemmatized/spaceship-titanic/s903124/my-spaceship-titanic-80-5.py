import pandas as pd
import numpy as np
import random as rnd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns = 999
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.describe()
_input1.info()
print()
print('--------' * 6)
_input0.info()
print(_input1.columns.values)
print('------' * 6)
print(_input1['HomePlanet'].value_counts())
print('------' * 6)
print(_input1['Destination'].value_counts())
print('------' * 6)
print(_input1['VIP'].value_counts())
print('------' * 6)
print(_input1['Transported'].value_counts())
print('------' * 6)
print(_input1['Cabin'].str[0].value_counts())
print('------' * 6)
print(_input1['CryoSleep'].value_counts())
print('------' * 6)
print(_input1['Age'].value_counts())
Missing_features = ['FoodCourt', 'Spa', 'ShoppingMall', 'RoomService', 'VRDeck', 'Cabin', 'CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Age']
for feature in Missing_features:
    if feature == 'Age':
        fill = _input1[feature].mean()
    else:
        fill = _input1[feature].value_counts().index[0]
    _input1[feature] = _input1[feature].fillna(fill)
    _input0[feature] = _input0[feature].fillna(fill)

def extract_deck(s):
    return s.split('/')[0]

def extract_num(s):
    return s.split('/')[1]

def extract_side(s):
    return s.split('/')[2]
_input1['Deck'] = _input1['Cabin'].apply(extract_deck)
_input1['Num'] = _input1['Cabin'].apply(extract_num)
_input1['Side'] = _input1['Cabin'].apply(extract_side)
_input0['Deck'] = _input0['Cabin'].apply(extract_deck)
_input0['Num'] = _input0['Cabin'].apply(extract_num)
_input0['Side'] = _input0['Cabin'].apply(extract_side)
_input1[['HomePlanet', 'Destination', 'Deck', 'Side']] = _input1[['HomePlanet', 'Destination', 'Deck', 'Side']].astype('category')
_input0[['HomePlanet', 'Destination', 'Deck', 'Side']] = _input0[['HomePlanet', 'Destination', 'Deck', 'Side']].astype('category')
data = pd.concat([_input1[_input0.columns], _input0])
data

def extract_last_name(s):
    return str(s).split(' ')[-1]
data['LastName'] = data['Name'].apply(extract_last_name)
dict_names = data['LastName'].value_counts().to_dict()

def same_name(s):
    return dict_names[s] - 1
data['SameName'] = data['LastName'].apply(same_name)
data
data = data.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name', 'Age', 'Deck', 'Side', 'LastName'], axis=1, inplace=False)
data[['CryoSleep', 'VIP', 'Num']] = data[['CryoSleep', 'VIP', 'Num']].astype(int)
X_train = data
y_train = _input1['Transported']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
xgb = XGBClassifier(gamma=1.5, subsample=1.0, max_depth=5, colsample_bytree=1.0, n_estimators=100)