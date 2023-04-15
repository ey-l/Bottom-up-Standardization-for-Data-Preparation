import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
PATH = 'data/input/spaceship-titanic/'

train_df = pd.read_csv(f'{PATH}train.csv', low_memory=False)
test_df = pd.read_csv(f'{PATH}test.csv', low_memory=False)
train_df.head()
train_df.tail()
train_df.isnull().sum().sort_values(ascending=False)
train_df.describe()
train_df[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['Cabin', 'Transported']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
g = sns.FacetGrid(train_df[train_df['Age'] > 0], col='Transported')
g.map(plt.hist, 'Age', bins=30)
grid = sns.FacetGrid(train_df, col='Transported', row='Destination', height=2.2, aspect=2)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df, row='VIP', col='Transported', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'CryoSleep', 'Destination', alpha=0.5, ci=None)
grid.add_legend()
train_df = train_df.drop(['PassengerId', 'Name'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name'], axis=1)
train_df.head()
train_df[['deck', 'num', 'side']] = train_df['Cabin'].str.split('/', expand=True)
train_df = train_df.drop(['Cabin'], axis=1)
train_df.head()
test_df[['deck', 'num', 'side']] = test_df['Cabin'].str.split('/', expand=True)
test_df = test_df.drop(['Cabin'], axis=1)
test_df.head()
train_df['deck'].value_counts()
train_df['deck'].unique().tolist()
train_df['deck'] = train_df['deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
train_df['deck'].value_counts()
test_df['deck'] = test_df['deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
test_df['deck'].value_counts()
train_df.head()
train_df['side'].unique().tolist()
train_df['side'] = train_df['side'].replace({'P': 0, 'S': 1})
train_df['side'].value_counts()
test_df['side'] = test_df['deck'].replace({'P': 0, 'S': 1})
test_df['side'].value_counts()
train_df.head()
train_df['HomePlanet'].unique().tolist()
train_df['HomePlanet'] = train_df['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 1})
train_df['HomePlanet'].value_counts()
test_df['HomePlanet'] = test_df['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 1})
test_df['HomePlanet'].value_counts()
train_df['Destination'].unique().tolist()
train_df['Destination'] = train_df['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
train_df['Destination'].value_counts()
test_df['Destination'] = test_df['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
test_df['Destination'].value_counts()
train_df.head()
train_df[['CryoSleep', 'VIP', 'Transported']] = (train_df[['CryoSleep', 'VIP', 'Transported']] == True).astype(int)
test_df[['CryoSleep', 'VIP']] = (test_df[['CryoSleep', 'VIP']] == True).astype(int)
train_df.tail()
nulls = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
train_df = train_df.select_dtypes(include=[np.number]).interpolate().dropna()
test_df = test_df.select_dtypes(include=[np.number]).interpolate().dropna()
nulls = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
X_train = train_df.drop('Transported', axis=1)
Y_train = train_df['Transported']
X_test = test_df
random_forest = RandomForestClassifier(n_estimators=100)