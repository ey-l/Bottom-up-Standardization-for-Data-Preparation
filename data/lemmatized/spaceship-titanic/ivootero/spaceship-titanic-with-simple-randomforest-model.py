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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.describe(include='all')
_input1 = pd.concat([_input1, _input1['Cabin'].str.split('/', expand=True)], axis=1)
_input1 = _input1.rename(columns={0: 'Cabin_Deck', 1: 'Cabin_Number', 2: 'Cabin_Side'}, inplace=False)
_input0 = pd.concat([_input0, _input0['Cabin'].str.split('/', expand=True)], axis=1)
_input0 = _input0.rename(columns={0: 'Cabin_Deck', 1: 'Cabin_Number', 2: 'Cabin_Side'}, inplace=False)
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Cabin_Number', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Cabin', 'Cabin_Number', 'Name'], axis=1, inplace=False)
_input1.sample(7)
sns.barplot(data=_input1, x='Transported', y=_input1.index, palette='Paired')
cat_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side']
(fig, axes) = plt.subplots(6, 2, figsize=(30, 35))
idx = 0
for col in cat_col:
    sns.countplot(data=_input1, y=col, palette='magma', orient='v', ax=axes[idx][0]).set_title(f'Count of {col}', fontsize='15')
    sns.countplot(data=_input1, y=col, palette='Paired', orient='v', hue='Transported', ax=axes[idx][1]).set_title(f'Count of {col} per transported', fontsize='15')
    idx += 1
_input1.isnull().sum()
len(_input1[_input1.isnull().all(axis=1)])
_input1 = _input1.fillna(value={'HomePlanet': 'Unknown'})
_input1 = _input1.fillna(value={'Destination': 'Unknown'})
_input1 = _input1.fillna(value={'CryoSleep': 'Unknown'})
_input0 = _input0.fillna(value={'HomePlanet': 'Unknown'})
_input0 = _input0.fillna(value={'Destination': 'Unknown'})
_input0 = _input0.fillna(value={'CryoSleep': 'Unknown'})
_input1.VIP = _input1.VIP.ffill()
_input1.Cabin_Deck = _input1.Cabin_Deck.ffill()
_input1.Cabin_Side = _input1.Cabin_Side.ffill()
_input0.VIP = _input0.VIP.ffill()
_input0.Cabin_Deck = _input0.Cabin_Deck.ffill()
_input0.Cabin_Side = _input0.Cabin_Side.ffill()
_input1.Age = _input1.Age.fillna(_input1.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].transform('mean')).round(0)
_input1.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].mean()
_input1.Age = _input1.Age.fillna(_input1['Age'].mean())
_input0.Age = _input0.Age.fillna(_input0.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].transform('mean')).round(0)
_input0.groupby(['HomePlanet', 'CryoSleep', 'VIP', 'Cabin_Side', 'Cabin_Deck'])['Age'].mean()
_input0.Age = _input0.Age.fillna(_input1['Age'].mean())
imputer_cols = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')