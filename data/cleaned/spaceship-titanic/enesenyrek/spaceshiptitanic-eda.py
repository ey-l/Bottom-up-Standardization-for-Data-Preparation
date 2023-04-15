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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head(10)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.head(10)
df_train.describe()
df_train.info()
df_train.dtypes
df_train[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df_train[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df_train[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df_train[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)

def bar_plot(variable):
    var = df_train[variable]
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
sns.countplot(y='Age', data=df_train, palette='cubehelix')
plt.xticks(rotation=90)

df_train['Cabin'].fillna('N/N/U', inplace=True)
df_test['Cabin'].fillna('N/N/U', inplace=True)
deck_num_side = df_train['Cabin'].apply(lambda x: x.split('/'))
side = list(map(lambda x: x[-1], deck_num_side))
side_order_vals = ['S', 'P', 'U']
df_train['side'] = side
df_test['side'] = list(map(lambda x: x[-1], df_test['Cabin'].apply(lambda x: x.split('/'))))
sns.catplot(x='side', kind='count', hue='Transported', data=df_train, palette='rocket').set(title='Cabin Side and Transported Count')

deck = list(map(lambda x: x[0], deck_num_side))
df_train['deck'] = deck
df_test['deck'] = list(map(lambda x: x[0], df_test['Cabin'].apply(lambda x: x.split('/'))))
deck_order_vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'N']
sns.catplot(x='deck', kind='count', hue='Transported', data=df_train, palette='mako').set(title='Cabin Deck and Transported Count')

sns.catplot(x='Destination', kind='count', hue='Transported', data=df_train, palette='rocket').set(title='Destination and Transported Count')

sns.catplot(x='CryoSleep', kind='count', hue='Transported', data=df_train, palette='rocket').set(title='CryoSleep and Transported Count')

sns.catplot(x='VIP', kind='count', hue='Transported', data=df_train, palette='rocket').set(title='VIP and Transported Count')

PASSENGER_ID = df_test[['PassengerId']]
df_train.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
missing_values = df_train.columns[df_train.isna().any()].tolist()
plt.figure(figsize=(19, 8))
nan_count_cols = df_train[missing_values].isna().sum()
print('Missing data in the train chart:')
sns.barplot(y=nan_count_cols, x=missing_values, palette='flare')

missing_values = df_test.columns[df_test.isna().any()].tolist()
plt.figure(figsize=(19, 8))
nan_count_cols = df_test[missing_values].isna().sum()
print('Missing data in the test chart:')
sns.barplot(y=nan_count_cols, x=missing_values, palette='rocket_r')

LABELS = df_test.columns
for col in LABELS:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df_train[col].fillna(df_train[col].median(), inplace=True)
        df_test[col].fillna(df_train[col].median(), inplace=True)
    else:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        df_test[col].fillna(df_train[col].mode()[0], inplace=True)
df_train[df_train['HomePlanet'].isnull()]
for col in LABELS:
    if df_train[col].dtype == 'O':
        encoder = LabelEncoder()
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
    elif df_train[col].dtype == 'bool':
        df_train[col] = df_train[col].astype('int')
        df_test[col] = df_test[col].astype('int')
encoder = LabelEncoder()
df_train['Transported'] = df_train['Transported'].astype('int')
LABELS_SCALE = ['Age']
scaler = MinMaxScaler()
df_train[LABELS_SCALE] = scaler.fit_transform(df_train[LABELS_SCALE])
df_test[LABELS_SCALE] = scaler.fit_transform(df_test[LABELS_SCALE])
df_train.info()
df_train.head(10)