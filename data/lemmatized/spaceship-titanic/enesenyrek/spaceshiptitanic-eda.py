import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(10)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head(10)
_input1.describe()
_input1.info()
_input1.dtypes
_input1[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)

def bar_plot(variable):
    var = _input1[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(9, 3))
    plt.bar(varValue.index, varValue, color='green')
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    print('{} \n {}'.format(variable, varValue))
category1 = ['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'Transported']
for c in category1:
    bar_plot(c)
plt.figure(figsize=(15, 10))
sns.countplot(y='Age', data=_input1, palette='cubehelix')
plt.xticks(rotation=90)
_input1['Cabin'] = _input1['Cabin'].fillna('N/N/U', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('N/N/U', inplace=False)
deck_num_side = _input1['Cabin'].apply(lambda x: x.split('/'))
side = list(map(lambda x: x[-1], deck_num_side))
side_order_vals = ['S', 'P', 'U']
_input1['side'] = side
_input0['side'] = list(map(lambda x: x[-1], _input0['Cabin'].apply(lambda x: x.split('/'))))
sns.catplot(x='side', kind='count', hue='Transported', data=_input1, palette='rocket').set(title='Cabin Side and Transported Count')
deck = list(map(lambda x: x[0], deck_num_side))
_input1['deck'] = deck
_input0['deck'] = list(map(lambda x: x[0], _input0['Cabin'].apply(lambda x: x.split('/'))))
deck_order_vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'N']
sns.catplot(x='deck', kind='count', hue='Transported', data=_input1, palette='mako').set(title='Cabin Deck and Transported Count')
sns.catplot(x='Destination', kind='count', hue='Transported', data=_input1, palette='rocket').set(title='Destination and Transported Count')
sns.catplot(x='CryoSleep', kind='count', hue='Transported', data=_input1, palette='rocket').set(title='CryoSleep and Transported Count')
sns.catplot(x='VIP', kind='count', hue='Transported', data=_input1, palette='rocket').set(title='VIP and Transported Count')
PASSENGER_ID = _input0[['PassengerId']]
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
missing_values = _input1.columns[_input1.isna().any()].tolist()
plt.figure(figsize=(19, 8))
nan_count_cols = _input1[missing_values].isna().sum()
print('Missing data in the train chart:')
sns.barplot(y=nan_count_cols, x=missing_values, palette='flare')
missing_values = _input0.columns[_input0.isna().any()].tolist()
plt.figure(figsize=(19, 8))
nan_count_cols = _input0[missing_values].isna().sum()
print('Missing data in the test chart:')
sns.barplot(y=nan_count_cols, x=missing_values, palette='rocket_r')
LABELS = _input0.columns
for col in LABELS:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
        _input0[col] = _input0[col].fillna(_input1[col].median(), inplace=False)
    else:
        _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
        _input0[col] = _input0[col].fillna(_input1[col].mode()[0], inplace=False)
_input1[_input1['HomePlanet'].isnull()]
for col in LABELS:
    if _input1[col].dtype == 'O':
        encoder = LabelEncoder()
        _input1[col] = encoder.fit_transform(_input1[col])
        _input0[col] = encoder.transform(_input0[col])
    elif _input1[col].dtype == 'bool':
        _input1[col] = _input1[col].astype('int')
        _input0[col] = _input0[col].astype('int')
encoder = LabelEncoder()
_input1['Transported'] = _input1['Transported'].astype('int')
LABELS_SCALE = ['Age']
scaler = MinMaxScaler()
_input1[LABELS_SCALE] = scaler.fit_transform(_input1[LABELS_SCALE])
_input0[LABELS_SCALE] = scaler.fit_transform(_input0[LABELS_SCALE])
_input1.info()
_input1.head(10)