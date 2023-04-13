import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy import stats
import os
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pID = _input1['PassengerId']
_input1.info()
print('train size:', _input1.shape)
print('test size:', _input0.shape)
_input1.head(100)
cats = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Name']

def show_nan(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
    missing_value_df = missing_value_df.sort_values('percent_missing', inplace=False)
    print(missing_value_df)
show_nan(_input1)
print()
_input1['RoomService'] = _input1['RoomService'].fillna(0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0)
_input1['Spa'] = _input1['Spa'].fillna(0)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0)
_input0['RoomService'] = _input0['RoomService'].fillna(0)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0)
_input0['Spa'] = _input0['Spa'].fillna(0)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0)
for i in _input1.columns:
    if _input1[i].isna().sum() > 0:
        if i not in cats:
            _input1[i] = _input1[i].fillna(_input1.groupby('Transported')[i].transform('mean'))
for i in _input0.columns:
    if _input0[i].isna().sum() > 0:
        if i not in cats:
            _input0[i] = _input0[i].fillna(_input0[i].mean())
_input1['Cabin'] = _input1['Cabin'].fillna(method='ffill')
_input0['Cabin'] = _input0['Cabin'].fillna(method='ffill')
_input1['deck'] = _input1['Cabin'].apply(lambda x: x.split('/')[0])
_input1['num'] = _input1['Cabin'].apply(lambda x: x.split('/')[1])
_input1['side'] = _input1['Cabin'].apply(lambda x: x.split('/')[2])
_input0['deck'] = _input0['Cabin'].apply(lambda x: x.split('/')[0])
_input0['num'] = _input0['Cabin'].apply(lambda x: x.split('/')[1])
_input0['side'] = _input0['Cabin'].apply(lambda x: x.split('/')[2])
del _input1['Cabin'], _input0['Cabin']
cats.remove('Cabin')
cats.append('deck')
cats.append('num')
cats.append('side')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
for i in _input0.columns:
    if _input0[i].isna().sum() > 0:
        if i in cats:
            _input0[i] = _input0[i].fillna(_input0[i].value_counts(ascending=True).index[-1])
cats.remove('Name')
_input1['group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0])
_input0['group'] = _input0['PassengerId'].apply(lambda x: x.split('_')[0])
_input1['Name'] = _input1['Name'].fillna(method='ffill')
_input0['Name'] = _input0['Name'].fillna(method='ffill')
temp = pd.DataFrame(_input1.groupby(['group'])['Name'])
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
_input1['has_relatives'] = _input1['group'].map(d)
temp = pd.DataFrame(_input0.groupby(['group'])['Name'])
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
_input0['has_relatives'] = _input0['group'].map(d)
print(_input1)
del _input1['Name'], _input1['group']
del _input0['Name'], _input0['group']
_input1['ttl_spnd'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['ttl_spnd'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1['Adult'] = True
_input1.loc[_input1['Age'] < 18, 'Adult'] = False
_input0['Adult'] = True
_input0.loc[_input0['Age'] < 18, 'Adult'] = False
print(cats)
_input0.head()
from sklearn.preprocessing import LabelEncoder
for i in cats:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((_input1[i], _input0[i])).astype(str)