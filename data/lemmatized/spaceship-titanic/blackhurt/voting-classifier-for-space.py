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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Earth': 0, 'Europa': 1, 'Mars': 2}, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}, inplace=False)
_input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1}, inplace=False)
_input1['Transported'] = _input1['Transported'].replace({False: 0, True: 1}, inplace=False)
_input1 = _input1.drop(['PassengerId'], axis=1, inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(0, inplace=False)
cabin = list(_input1['Cabin'])
a = []
b = []
for i in cabin:
    if i != 0:
        a.append(i[0])
        b.append(i[-1])
    else:
        a.append(np.nan)
        b.append(np.nan)
_input1['Cabin_group1'] = a
_input1['Cabin_group2'] = b
_input1 = _input1.drop(['Cabin', 'Name'], axis=1, inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Earth': 0, 'Europa': 1, 'Mars': 2}, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=False)
_input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1}, inplace=False)
_input1['Transported'] = _input1['Transported'].replace({True: 0, False: 1}, inplace=False)
_input1['Cabin_group1'] = _input1['Cabin_group1'].replace({'F': 0, 'G': 1, 'E': 2, 'B': 3, 'C': 4, 'D': 5, 'A': 6, 'T': 7}, inplace=False)
_input1['Cabin_group2'] = _input1['Cabin_group2'].replace({'S': 0, 'P': 1}, inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1.groupby('HomePlanet')['Age'].transform('mean'))
_input1['RoomService'] = _input1['RoomService'].fillna(_input1.groupby('CryoSleep')['RoomService'].transform('mean'))
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1.groupby('Cabin_group1')['FoodCourt'].transform('mean'))
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1.groupby('CryoSleep')['ShoppingMall'].transform('mean'))
_input1['Spa'] = _input1['Spa'].fillna(_input1.groupby('CryoSleep')['Spa'].transform('mean'))
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1.groupby('Cabin_group1')['VRDeck'].transform('mean'))
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='most_frequent')