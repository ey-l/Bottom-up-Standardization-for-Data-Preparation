import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import scipy as sc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
survived_age = train[train['Transported'] == False]['Age']
transported_age = train[train['Transported'] == True]['Age']
fig = plt.figure(figsize=(16, 8))
plt.hist(survived_age, bins=np.arange(0, 85, 5), density=True, label='Survived', alpha=0.5, color='#3c9db0')
plt.hist(transported_age, bins=np.arange(0, 85, 5), density=True, label='Transported', alpha=0.35, color='#D7503C')
plt.legend(frameon=False, fontsize='large')

train_copy = train.copy()
train_copy[['deck', 'num', 'side']] = train_copy['Cabin'].str.split('/', expand=True)
train_copy['HomePlanet'] = train_copy['HomePlanet'].astype('category').cat.codes
train_copy['Destination'] = train_copy['Destination'].astype('category').cat.codes
train_copy['deck'] = train_copy['deck'].astype('category').cat.codes
train_copy['side'] = train_copy['side'].astype('category').cat.codes
train_copy.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
sns.set(rc={'figure.figsize': (16, 8)})
sns.heatmap(train_copy.corr(), annot=True, fmt='.2g', cmap='coolwarm')
train[['deck', 'num', 'side']] = train['Cabin'].str.split('/', expand=True)
columns = ['CryoSleep', 'Age', 'HomePlanet', 'Spa', 'VRDeck', 'RoomService', 'Destination', 'deck', 'num', 'side', 'Transported']
train_cleaned = train[columns].dropna()