import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.info()
train_df.describe(include='all')
train_df = pd.concat([train_df, train_df['Cabin'].str.split('/', expand=True)], axis=1)
train_df.rename(columns={0: 'Cabin_Deck', 1: 'Cabin_Number', 2: 'Cabin_Side'}, inplace=True)
test_df = pd.concat([test_df, test_df['Cabin'].str.split('/', expand=True)], axis=1)
test_df.rename(columns={0: 'Cabin_Deck', 1: 'Cabin_Number', 2: 'Cabin_Side'}, inplace=True)
train_df.drop(['PassengerId', 'Cabin', 'Cabin_Number', 'Name'], axis=1, inplace=True)
test_df.drop(['Cabin', 'Cabin_Number', 'Name'], axis=1, inplace=True)
train_df.sample(7)
sns.barplot(data=train_df, x='Transported', y=train_df.index, palette='Paired')
cat_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']
(fig, axes) = plt.subplots(6, 2, figsize=(30, 35))
idx = 0
for col in cat_col:
    sns.countplot(data=train_df, y=col, palette='magma', orient='v', ax=axes[idx][0]).set_title(f'Count of {col}', fontsize='15')
    sns.countplot(data=train_df, y=col, palette='Paired', orient='v', hue='Transported', ax=axes[idx][1]).set_title(f'Count of {col} per transported', fontsize='15')
    idx += 1

train_df.isnull().sum()
len(train_df[train_df.isnull().all(axis=1)])
train_df = train_df.fillna(value={'HomePlanet': 'Unknown'})
train_df = train_df.fillna(value={'Destination': 'Unknown'})
train_df = train_df.fillna(value={'CryoSleep': 'Unknown'})
test_df = test_df.fillna(value={'HomePlanet': 'Unknown'})
test_df = test_df.fillna(value={'Destination': 'Unknown'})
test_df = test_df.fillna(value={'CryoSleep': 'Unknown'})
train_df.VIP = train_df.VIP.ffill()
train_df.Cabin_Deck = train_df.Cabin_Deck.ffill()
train_df.Cabin_Side = train_df.Cabin_Side.ffill()
test_df.VIP = test_df.VIP.ffill()
test_df.Cabin_Deck = test_df.Cabin_Deck.ffill()
test_df.Cabin_Side = test_df.Cabin_Side.ffill()
train_df.Age = train_df.Age.fillna(train_df.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].transform('mean')).round(0)
train_df.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].mean()
train_df.Age = train_df.Age.fillna(train_df['Age'].mean())
test_df.Age = test_df.Age.fillna(test_df.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].transform('mean')).round(0)
test_df.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].mean()
test_df.Age = test_df.Age.fillna(train_df['Age'].mean())
imputer_cols = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')