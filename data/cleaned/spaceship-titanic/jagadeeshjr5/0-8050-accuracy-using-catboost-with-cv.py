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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.head()
print(train_df.shape)
print(test_df.shape)
print(train_df.info())
train_df.isna().sum()
test_df.isna().sum()
train_df = train_df.drop('Name', axis=1)
test_df = test_df.drop('Name', axis=1)
train_df.head()
count = 0
for i in range(len(train_df['HomePlanet'])):
    if str(train_df['HomePlanet'][i]) == 'nan':
        n = train_df['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in train_df['PassengerId'].values:
            train_df['HomePlanet'][i] = train_df[train_df['PassengerId'] == n1]['HomePlanet'].tolist()[0]
            count = count + 1
        elif n2 in train_df['PassengerId'].values:
            train_df['HomePlanet'][i] = train_df[train_df['PassengerId'] == n2]['HomePlanet'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(test_df['HomePlanet'])):
    if str(test_df['HomePlanet'][i]) == 'nan':
        n = test_df['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in test_df['PassengerId'].values:
            test_df['HomePlanet'][i] = test_df[test_df['PassengerId'] == n1]['HomePlanet'].tolist()[0]
            count = count + 1
        elif n2 in test_df['PassengerId'].values:
            test_df['HomePlanet'][i] = test_df[test_df['PassengerId'] == n2]['HomePlanet'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(train_df['Cabin'])):
    if str(train_df['Cabin'][i]) == 'nan':
        n = train_df['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in train_df['PassengerId'].values:
            train_df['Cabin'][i] = train_df[train_df['PassengerId'] == n1]['Cabin'].tolist()[0]
            count = count + 1
        elif n2 in train_df['PassengerId'].values:
            train_df['Cabin'][i] = train_df[train_df['PassengerId'] == n2]['Cabin'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(test_df['Cabin'])):
    if str(test_df['Cabin'][i]) == 'nan':
        n = test_df['PassengerId'][i]
        n1 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) - 1)
        n2 = n.split('_')[0] + '_0' + str(int(n.split('_')[1]) + 1)
        if n1 in test_df['PassengerId'].values:
            test_df['Cabin'][i] = test_df[test_df['PassengerId'] == n1]['Cabin'].tolist()[0]
            count = count + 1
        elif n2 in test_df['PassengerId'].values:
            test_df['Cabin'][i] = test_df[test_df['PassengerId'] == n2]['Cabin'].tolist()[0]
            count = count + 1
        else:
            continue
print(count, 'Values Imputed')
count = 0
for i in range(len(train_df['CryoSleep'])):
    if str(train_df['CryoSleep'][i]) == 'nan':
        if train_df['RoomService'][i] == 0.0:
            train_df['CryoSleep'][i] = True
            count = count + 1
        else:
            train_df['CryoSleep'][i] = False
            count = count + 1
print(count, 'Values Imputed')
count = 0
for i in range(len(test_df['CryoSleep'])):
    if str(test_df['CryoSleep'][i]) == 'nan':
        if test_df['RoomService'][i] == 0.0:
            test_df['CryoSleep'][i] = True
            count = count + 1
        else:
            test_df['CryoSleep'][i] = False
            count = count + 1
print(count, 'Values Imputed')
for i in range(len(train_df['RoomService'])):
    if str(train_df['RoomService'][i]) == 'nan':
        if train_df['CryoSleep'][i] == 'True':
            train_df['RoomService'][i] = 0.0
        else:
            train_df['RoomService'][i] = 1.0
for i in range(len(test_df['RoomService'])):
    if str(test_df['RoomService'][i]) == 'nan':
        if test_df['CryoSleep'][i] == 'True':
            test_df['RoomService'][i] = 0.0
        else:
            test_df['RoomService'][i] = 1.0
for i in range(len(train_df['FoodCourt'])):
    if str(train_df['FoodCourt'][i]) == 'nan':
        if train_df['CryoSleep'][i] == 'True':
            train_df['FoodCourt'][i] = 0.0
        else:
            train_df['FoodCourt'][i] = 1.0
for i in range(len(test_df['FoodCourt'])):
    if str(test_df['FoodCourt'][i]) == 'nan':
        if test_df['CryoSleep'][i] == 'True':
            test_df['FoodCourt'][i] = 0.0
        else:
            test_df['FoodCourt'][i] = 1.0
for i in range(len(train_df['ShoppingMall'])):
    if str(train_df['ShoppingMall'][i]) == 'nan':
        if train_df['CryoSleep'][i] == 'True':
            train_df['ShoppingMall'][i] = 0.0
        else:
            train_df['ShoppingMall'][i] = 1.0
for i in range(len(test_df['ShoppingMall'])):
    if str(test_df['ShoppingMall'][i]) == 'nan':
        if test_df['CryoSleep'][i] == 'True':
            test_df['ShoppingMall'][i] = 0.0
        else:
            test_df['ShoppingMall'][i] = 1.0
for i in range(len(train_df['Spa'])):
    if str(train_df['Spa'][i]) == 'nan':
        if train_df['CryoSleep'][i] == 'True':
            train_df['Spa'][i] = 0.0
        else:
            train_df['Spa'][i] = 1.0
for i in range(len(test_df['Spa'])):
    if str(test_df['Spa'][i]) == 'nan':
        if test_df['CryoSleep'][i] == 'True':
            test_df['Spa'][i] = 0.0
        else:
            test_df['Spa'][i] = 1.0
train_df['VIP'].fillna(value=False, inplace=True)
test_df['VIP'].fillna(value=False, inplace=True)
for i in range(len(train_df['VRDeck'])):
    if str(train_df['VRDeck'][i]) == 'nan':
        if train_df['CryoSleep'][i] == 'True':
            train_df['VRDeck'][i] = 0.0
        else:
            train_df['VRDeck'][i] = float(round(train_df['VRDeck'].mean()))
for i in range(len(test_df['VRDeck'])):
    if str(test_df['VRDeck'][i]) == 'nan':
        if test_df['CryoSleep'][i] == 'True':
            test_df['VRDeck'][i] = 0.0
        else:
            test_df['VRDeck'][i] = float(round(test_df['VRDeck'].mean()))
train_df['Age'].fillna(value=train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(value=test_df['Age'].median(), inplace=True)
test_df['HomePlanet'].fillna(value=test_df['HomePlanet'].mode()[0], inplace=True)
test_df['Destination'].fillna(value=test_df['Destination'].mode()[0], inplace=True)
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)
train_df['VIP'] = train_df['VIP'].astype(int)
test_df['VIP'] = test_df['VIP'].astype(int)
train_df['Transported'] = train_df['Transported'].astype(int)
train_df.head(3)
train_df.dropna(axis=0, inplace=True)
print(train_df.isna().sum())
print(test_df.isna().sum())
print(train_df.shape)
print(test_df.shape)
train_df = train_df.drop(['PassengerId', 'Cabin'], axis='columns')
test_df = test_df.drop(['PassengerId', 'Cabin'], axis='columns')
categorical_cols = ['HomePlanet', 'Destination']
dummies = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
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