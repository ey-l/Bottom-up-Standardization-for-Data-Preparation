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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
train.describe()
train.info()
print()
print('--------' * 6)
test.info()
print(train.columns.values)
print('------' * 6)
print(train['HomePlanet'].value_counts())
print('------' * 6)
print(train['Destination'].value_counts())
print('------' * 6)
print(train['VIP'].value_counts())
print('------' * 6)
print(train['Transported'].value_counts())
print('------' * 6)
print(train['Cabin'].str[0].value_counts())
print('------' * 6)
print(train['CryoSleep'].value_counts())
print('------' * 6)
print(train['Age'].value_counts())
Missing_features = ['FoodCourt', 'Spa', 'ShoppingMall', 'RoomService', 'VRDeck', 'Cabin', 'CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Age']
for feature in Missing_features:
    if feature == 'Age':
        fill = train[feature].mean()
    else:
        fill = train[feature].value_counts().index[0]
    train[feature] = train[feature].fillna(fill)
    test[feature] = test[feature].fillna(fill)

def extract_deck(s):
    return s.split('/')[0]

def extract_num(s):
    return s.split('/')[1]

def extract_side(s):
    return s.split('/')[2]
train['Deck'] = train['Cabin'].apply(extract_deck)
train['Num'] = train['Cabin'].apply(extract_num)
train['Side'] = train['Cabin'].apply(extract_side)
test['Deck'] = test['Cabin'].apply(extract_deck)
test['Num'] = test['Cabin'].apply(extract_num)
test['Side'] = test['Cabin'].apply(extract_side)
train[['HomePlanet', 'Destination', 'Deck', 'Side']] = train[['HomePlanet', 'Destination', 'Deck', 'Side']].astype('category')
test[['HomePlanet', 'Destination', 'Deck', 'Side']] = test[['HomePlanet', 'Destination', 'Deck', 'Side']].astype('category')
data = pd.concat([train[test.columns], test])
data

def extract_last_name(s):
    return str(s).split(' ')[-1]
data['LastName'] = data['Name'].apply(extract_last_name)
dict_names = data['LastName'].value_counts().to_dict()

def same_name(s):
    return dict_names[s] - 1
data['SameName'] = data['LastName'].apply(same_name)
data
data.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'Name', 'Age', 'Deck', 'Side', 'LastName'], axis=1, inplace=True)
data[['CryoSleep', 'VIP', 'Num']] = data[['CryoSleep', 'VIP', 'Num']].astype(int)
X_train = data
y_train = train['Transported']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
xgb = XGBClassifier(gamma=1.5, subsample=1.0, max_depth=5, colsample_bytree=1.0, n_estimators=100)