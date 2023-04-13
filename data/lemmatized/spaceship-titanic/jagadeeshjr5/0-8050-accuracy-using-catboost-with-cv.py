import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
print(_input1.shape)
print(_input0.shape)
print(_input1.info())
_input1.isna().sum()
_input0.isna().sum()
_input1 = _input1.drop('Name', axis=1)
_input0 = _input0.drop('Name', axis=1)
_input1.head()
count = 0
for i in range(len(_input1['HomePlanet'])):
    if str(_input1['HomePlanet'][i]) == 'nan':
        n = _input1['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in _input1['PassengerId'].values:
            _input1['HomePlanet'][i] = _input1[_input1['PassengerId'] == n1]['HomePlanet'].tolist()[0]
            count = count + 1
        elif n2 in _input1['PassengerId'].values:
            _input1['HomePlanet'][i] = _input1[_input1['PassengerId'] == n2]['HomePlanet'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(_input0['HomePlanet'])):
    if str(_input0['HomePlanet'][i]) == 'nan':
        n = _input0['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in _input0['PassengerId'].values:
            _input0['HomePlanet'][i] = _input0[_input0['PassengerId'] == n1]['HomePlanet'].tolist()[0]
            count = count + 1
        elif n2 in _input0['PassengerId'].values:
            _input0['HomePlanet'][i] = _input0[_input0['PassengerId'] == n2]['HomePlanet'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(_input1['Cabin'])):
    if str(_input1['Cabin'][i]) == 'nan':
        n = _input1['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in _input1['PassengerId'].values:
            _input1['Cabin'][i] = _input1[_input1['PassengerId'] == n1]['Cabin'].tolist()[0]
            count = count + 1
        elif n2 in _input1['PassengerId'].values:
            _input1['Cabin'][i] = _input1[_input1['PassengerId'] == n2]['Cabin'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(_input0['Cabin'])):
    if str(_input0['Cabin'][i]) == 'nan':
        n = _input0['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in _input0['PassengerId'].values:
            _input0['Cabin'][i] = _input0[_input0['PassengerId'] == n1]['Cabin'].tolist()[0]
            count = count + 1
        elif n2 in _input0['PassengerId'].values:
            _input0['Cabin'][i] = _input0[_input0['PassengerId'] == n2]['Cabin'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(_input1['CryoSleep'])):
    if str(_input1['CryoSleep'][i]) == 'nan':
        if _input1['RoomService'][i] == 0.0:
            _input1['CryoSleep'][i] = True
            count = count + 1
        else:
            _input1['CryoSleep'][i] = False
            count = count + 1
print(count, 'Values Imputed')
count = 0
for i in range(len(_input0['CryoSleep'])):
    if str(_input0['CryoSleep'][i]) == 'nan':
        if _input0['RoomService'][i] == 0.0:
            _input0['CryoSleep'][i] = True
            count = count + 1
        else:
            _input0['CryoSleep'][i] = False
            count = count + 1
print(count, 'Values Imputed')
for i in range(len(_input1['RoomService'])):
    if str(_input1['RoomService'][i]) == 'nan':
        if _input1['CryoSleep'][i] == 'True':
            _input1['RoomService'][i] = 0.0
        else:
            _input1['RoomService'][i] = 1.0
for i in range(len(_input0['RoomService'])):
    if str(_input0['RoomService'][i]) == 'nan':
        if _input0['CryoSleep'][i] == 'True':
            _input0['RoomService'][i] = 0.0
        else:
            _input0['RoomService'][i] = 1.0
for i in range(len(_input1['FoodCourt'])):
    if str(_input1['FoodCourt'][i]) == 'nan':
        if _input1['CryoSleep'][i] == 'True':
            _input1['FoodCourt'][i] = 0.0
        else:
            _input1['FoodCourt'][i] = 1.0
for i in range(len(_input0['FoodCourt'])):
    if str(_input0['FoodCourt'][i]) == 'nan':
        if _input0['CryoSleep'][i] == 'True':
            _input0['FoodCourt'][i] = 0.0
        else:
            _input0['FoodCourt'][i] = 1.0
for i in range(len(_input1['ShoppingMall'])):
    if str(_input1['ShoppingMall'][i]) == 'nan':
        if _input1['CryoSleep'][i] == 'True':
            _input1['ShoppingMall'][i] = 0.0
        else:
            _input1['ShoppingMall'][i] = 1.0
for i in range(len(_input0['ShoppingMall'])):
    if str(_input0['ShoppingMall'][i]) == 'nan':
        if _input0['CryoSleep'][i] == 'True':
            _input0['ShoppingMall'][i] = 0.0
        else:
            _input0['ShoppingMall'][i] = 1.0
for i in range(len(_input1['Spa'])):
    if str(_input1['Spa'][i]) == 'nan':
        if _input1['CryoSleep'][i] == 'True':
            _input1['Spa'][i] = 0.0
        else:
            _input1['Spa'][i] = 1.0
for i in range(len(_input0['Spa'])):
    if str(_input0['Spa'][i]) == 'nan':
        if _input0['CryoSleep'][i] == 'True':
            _input0['Spa'][i] = 0.0
        else:
            _input0['Spa'][i] = 1.0
_input1['VIP'] = _input1['VIP'].fillna(value=False, inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(value=False, inplace=False)
for i in range(len(_input1['VRDeck'])):
    if str(_input1['VRDeck'][i]) == 'nan':
        if _input1['CryoSleep'][i] == 'True':
            _input1['VRDeck'][i] = 0.0
        else:
            _input1['VRDeck'][i] = float(round(_input1['VRDeck'].mean()))
for i in range(len(_input0['VRDeck'])):
    if str(_input0['VRDeck'][i]) == 'nan':
        if _input0['CryoSleep'][i] == 'True':
            _input0['VRDeck'][i] = 0.0
        else:
            _input0['VRDeck'][i] = float(round(_input0['VRDeck'].mean()))
_input1['Age'] = _input1['Age'].fillna(value=_input1['Age'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(value=_input0['Age'].median(), inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(value=_input0['HomePlanet'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(value=_input0['Destination'].mode()[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
_input1['VIP'] = _input1['VIP'].astype(int)
_input0['VIP'] = _input0['VIP'].astype(int)
_input1['Transported'] = _input1['Transported'].astype(int)
_input1.head(3)
_input1 = _input1.dropna(axis=0, inplace=False)
print(_input1.isna().sum())
print(_input0.isna().sum())
print(_input1.shape)
print(_input0.shape)
_input1 = _input1.drop(['PassengerId', 'Cabin'], axis='columns')
_input0 = _input0.drop(['PassengerId', 'Cabin'], axis='columns')
categorical_cols = ['HomePlanet', 'Destination']
dummies = pd.get_dummies(_input1, columns=categorical_cols, drop_first=True)
_input0 = pd.get_dummies(_input0, columns=categorical_cols, drop_first=True)
dummies.columns
test = dummies['Transported']
dummmies = dummies.drop('Transported', axis='columns')
train = dummmies
print(train.shape)
print(test.shape)
train.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train, test, test_size=0.2, random_state=42)
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
Log_reg = LogisticRegression()
params = {'C': [0.01, 0.1, 0.5, 0.001, 1]}
model1 = GridSearchCV(Log_reg, params, cv=5)