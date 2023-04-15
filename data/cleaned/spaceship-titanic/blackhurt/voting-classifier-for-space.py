import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train['HomePlanet'].replace({'Earth': 0, 'Europa': 1, 'Mars': 2}, inplace=True)
train['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
train['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}, inplace=True)
train['VIP'].replace({False: 0, True: 1}, inplace=True)
train['Transported'].replace({False: 0, True: 1}, inplace=True)
train.drop(['PassengerId'], axis=1, inplace=True)
train['Cabin'].fillna(0, inplace=True)
cabin = list(train['Cabin'])
a = []
b = []
for i in cabin:
    if i != 0:
        a.append(i[0])
        b.append(i[-1])
    else:
        a.append(np.nan)
        b.append(np.nan)
train['Cabin_group1'] = a
train['Cabin_group2'] = b
train.drop(['Cabin', 'Name'], axis=1, inplace=True)
train['HomePlanet'].replace({'Earth': 0, 'Europa': 1, 'Mars': 2}, inplace=True)
train['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
train['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=True)
train['VIP'].replace({False: 0, True: 1}, inplace=True)
train['Transported'].replace({True: 0, False: 1}, inplace=True)
train['Cabin_group1'].replace({'F': 0, 'G': 1, 'E': 2, 'B': 3, 'C': 4, 'D': 5, 'A': 6, 'T': 7}, inplace=True)
train['Cabin_group2'].replace({'S': 0, 'P': 1}, inplace=True)
train['Age'] = train['Age'].fillna(train.groupby('HomePlanet')['Age'].transform('mean'))
train['RoomService'] = train['RoomService'].fillna(train.groupby('CryoSleep')['RoomService'].transform('mean'))
train['FoodCourt'] = train['FoodCourt'].fillna(train.groupby('Cabin_group1')['FoodCourt'].transform('mean'))
train['ShoppingMall'] = train['ShoppingMall'].fillna(train.groupby('CryoSleep')['ShoppingMall'].transform('mean'))
train['Spa'] = train['Spa'].fillna(train.groupby('CryoSleep')['Spa'].transform('mean'))
train['VRDeck'] = train['VRDeck'].fillna(train.groupby('Cabin_group1')['VRDeck'].transform('mean'))
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='most_frequent')