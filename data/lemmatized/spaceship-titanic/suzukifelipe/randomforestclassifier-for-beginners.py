import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.describe()
_input1.info()
_input0.info()
df = pd.concat([_input1, _input0])
df.info()
df.isna().any(axis=1).sum()
print('The total of rows with missing values is: ', round(df.isna().any(axis=1).sum() / len(df.index) * 100, 2), '%')
cols_to_replace = {'HomePlanet': 'Unknown', 'Destination': 'Unknown', 'Name': 'Unknown'}
df = df.fillna(value=cols_to_replace, inplace=False)
df.info()
df = df.fillna({'Cabin': 'U/0/U'}, inplace=False)
df = df.fillna({'CryoSleep': 9}, inplace=False)
df['CryoSleep'] = df['CryoSleep'].astype(int)
df['CryoSleep'].value_counts()
df = df.fillna({'VIP': 9}, inplace=False)
df['VIP'] = df['VIP'].astype(int)
df['VIP'].value_counts()
cols_median = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[cols_median] = df[cols_median].fillna(df[cols_median].median())
df['PassengerId'].value_counts()
group = df['PassengerId'].str[:4]
df.insert(loc=1, column='GroupId', value=group)
df.head()
deck = df['Cabin'].str[:1]
deck.value_counts()
df.insert(loc=5, column='CabinDeck', value=deck)
df.head()
deck_sub = {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
df = df.replace({'CabinDeck': deck_sub}, inplace=False)
df.head()
num = df['Cabin'].str[2:-2]
side = df['Cabin'].str[-1:]
df.insert(loc=6, column='CabinNum', value=num)
df.insert(loc=7, column='CabinSide', value=side)
df['HomePlanet'].value_counts()
df['Destination'].value_counts()
homeplanet = {'Unknown': 0, 'Earth': 1, 'Europa': 2, 'Mars': 3}
cab_side = {'U': 0, 'P': 1, 'S': 2}
dest = {'Unknown': 0, 'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3}
df = df.replace({'HomePlanet': homeplanet, 'CabinSide': cab_side, 'Destination': dest}, inplace=False)
df['GroupId'] = df['GroupId'].astype('int')
df['CabinNum'] = df['CabinNum'].astype('int')
df['CabinSide'] = df['CabinSide'].astype('int')
lname = df['Name'].str.split(' ').str[-1]
df.insert(loc=17, column='Lastname', value=lname)
df['Lastname'].value_counts()
relatives = df['Lastname'].map(df['Lastname'].value_counts())
relatives = relatives - 1
relatives = relatives.replace({293: 0}, inplace=False)
df.insert(loc=18, column='Relatives', value=relatives)
df['Relatives'] = df['Relatives'].astype('int')
train_df = df.iloc[:8693, :]
test_df = df.iloc[8693:, :]
train_df.info()
test_df.info()
train_df['Transported'].value_counts()
train_df['Transported'] = train_df['Transported'].astype('int64')
train_df.info()
corr = train_df.corr()
corr['Transported'].sort_values(ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, cmap='Blues', annot=True, linewidths=0.5, ax=ax)
features = train_df.drop(['PassengerId', 'Cabin', 'Name', 'Lastname', 'Transported'], axis=1)
labels = train_df['Transported']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, features, labels, cv=10)
forest_scores.mean()