import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.isnull().sum()
column = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    print(column[i].value_counts())
column = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in column:
    _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
    _input0[col] = _input0[col].fillna(_input0[col].mode()[0], inplace=False)
    print(col)
sns.distplot(x=_input1['Age'])
sns.distplot(x=_input0['Age'])
column = ['Age']
for i in column:
    _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
    _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
    print(i)
Spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Spendings:
    _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
    _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
    print(i)
_input1 = _input1.drop(columns=['Cabin', 'Name'], axis=1)
_input0 = _input0.drop(columns=['Cabin', 'Name'], axis=1)
_input1.head()
_input0.head()
_input1[['CryoSleep', 'VIP']] = _input1[['CryoSleep', 'VIP']].astype('int')
_input0[['CryoSleep', 'VIP']] = _input0[['CryoSleep', 'VIP']].astype('int')
_input1.head()
_input0.head()
cat_var = _input1[['HomePlanet', 'Destination']]
num_val = pd.get_dummies(cat_var)
cat_var_test = _input0[['HomePlanet', 'Destination']]
num_val_test = pd.get_dummies(cat_var_test)
num_val_test.head()
num_val.head()
_input1 = _input1.drop(columns=['HomePlanet', 'Destination'])
_input0 = _input0.drop(columns=['HomePlanet', 'Destination'])
_input1 = pd.concat([_input1, num_val], axis=1)
_input0 = pd.concat([_input0, num_val_test], axis=1)
X = _input1.drop(columns=['Transported'], axis=1)
y = _input1['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(max_depth=9, random_state=0)