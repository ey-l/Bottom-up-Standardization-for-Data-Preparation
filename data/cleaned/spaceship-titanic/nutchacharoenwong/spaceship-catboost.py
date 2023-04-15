import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy import stats
import os

train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
pID = train['PassengerId']
train.info()
print('train size:', train.shape)
print('test size:', test.shape)
train.head(100)
cats = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Name']

def show_nan(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    print(missing_value_df)
show_nan(train)
print()
train['RoomService'] = train['RoomService'].fillna(0)
train['FoodCourt'] = train['FoodCourt'].fillna(0)
train['ShoppingMall'] = train['ShoppingMall'].fillna(0)
train['Spa'] = train['Spa'].fillna(0)
train['VRDeck'] = train['VRDeck'].fillna(0)
test['RoomService'] = test['RoomService'].fillna(0)
test['FoodCourt'] = test['FoodCourt'].fillna(0)
test['ShoppingMall'] = test['ShoppingMall'].fillna(0)
test['Spa'] = test['Spa'].fillna(0)
test['VRDeck'] = test['VRDeck'].fillna(0)
for i in train.columns:
    if train[i].isna().sum() > 0:
        if i not in cats:
            train[i] = train[i].fillna(train.groupby('Transported')[i].transform('mean'))
for i in test.columns:
    if test[i].isna().sum() > 0:
        if i not in cats:
            test[i] = test[i].fillna(test[i].mean())
train['Cabin'] = train['Cabin'].fillna(method='ffill')
test['Cabin'] = test['Cabin'].fillna(method='ffill')
train['deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['num'] = train['Cabin'].apply(lambda x: x.split('/')[1])
train['side'] = train['Cabin'].apply(lambda x: x.split('/')[2])
test['deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['num'] = test['Cabin'].apply(lambda x: x.split('/')[1])
test['side'] = test['Cabin'].apply(lambda x: x.split('/')[2])
del train['Cabin'], test['Cabin']
cats.remove('Cabin')
cats.append('deck')
cats.append('num')
cats.append('side')
train['CryoSleep'] = train['CryoSleep'].fillna(False)
test['CryoSleep'] = test['CryoSleep'].fillna(False)
for i in test.columns:
    if test[i].isna().sum() > 0:
        if i in cats:
            test[i] = test[i].fillna(test[i].value_counts(ascending=True).index[-1])
cats.remove('Name')
train['group'] = train['PassengerId'].apply(lambda x: x.split('_')[0])
test['group'] = test['PassengerId'].apply(lambda x: x.split('_')[0])
train['Name'] = train['Name'].fillna(method='ffill')
test['Name'] = test['Name'].fillna(method='ffill')
temp = pd.DataFrame(train.groupby(['group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
train['has_relatives'] = train['group'].map(d)
temp = pd.DataFrame(test.groupby(['group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
test['has_relatives'] = test['group'].map(d)
print(train)
del train['Name'], train['group']
del test['Name'], test['group']
train['ttl_spnd'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
test['ttl_spnd'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
train['Adult'] = True
train.loc[train['Age'] < 18, 'Adult'] = False
test['Adult'] = True
test.loc[test['Age'] < 18, 'Adult'] = False
print(cats)
test.head()
from sklearn.preprocessing import LabelEncoder
for i in cats:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((train[i], test[i])).astype(str)