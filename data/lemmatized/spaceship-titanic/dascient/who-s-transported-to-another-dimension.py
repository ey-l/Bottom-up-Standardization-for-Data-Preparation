import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.loc[_input1.Cabin.isnull(), 'Cabin'] = 'Z/9999/Z'
print('All passengers in CryoSleep & VIP were transported:')
all(_input1.loc[(_input1['CryoSleep'] == True) & (_input1['VIP'] == True)].Transported == True)
list0 = ['F', 'G', 'E', 'B', 'D', 'A', 'T']
list1 = ['S', 'P']
for (i, v) in _input1.Cabin.items():
    cabin = str(v).split('/')
    try:
        _input1.at[i, 'Cabin_x'] = cabin[0]
    except:
        _input1.at[i, 'Cabin_x'] = list0[random.randint(0, 6)]
    try:
        _input1.at[i, 'Cabin_y'] = int(cabin[1])
    except:
        _input1.at[i, 'Cabin_y'] = random.randint(0, 1894)
    try:
        _input1.at[i, 'Cabin_z'] = cabin[2]
    except:
        _input1.at[i, 'Cabin_z'] = list1[random.randint(0, 1)]
for (i, v) in _input0.Cabin.items():
    cabin = str(v).split('/')
    try:
        _input0.at[i, 'Cabin_x'] = cabin[0]
    except:
        _input0.at[i, 'Cabin_x'] = list0[random.randint(0, 6)]
    try:
        _input0.at[i, 'Cabin_y'] = int(cabin[1])
    except:
        _input0.at[i, 'Cabin_y'] = random.randint(0, 1894)
    try:
        _input0.at[i, 'Cabin_z'] = cabin[2]
    except:
        _input0.at[i, 'Cabin_z'] = list1[random.randint(0, 1)]
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].median(), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
_input1['Name'] = _input1['Name'].fillna('John Doe', inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(True, inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1.info()
print(len(_input1.loc[_input1.Cabin == 'Z/9999/Z']), 'null cabins (i.e. missing features)')
cabin_train = _input1[['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Cabin_x', 'Cabin_y', 'Cabin_z']]
cabin_test = _input1.loc[_input1.Cabin == 'Z/9999/Z'][['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
lb_make = LabelEncoder()
train_ml = cabin_train.drop(columns=['Cabin_x', 'Cabin_y', 'Cabin_z']).copy()
test_ml = cabin_test.copy()
train_ml['HomePlanet'] = lb_make.fit_transform(train_ml['HomePlanet'])
train_ml['Destination'] = lb_make.fit_transform(train_ml['Destination'])
train_ml['CryoSleep'] = lb_make.fit_transform(train_ml['CryoSleep'])
train_ml['VIP'] = lb_make.fit_transform(train_ml['VIP'])
train_ml['Transported'] = lb_make.fit_transform(train_ml['Transported'])
test_ml['HomePlanet'] = lb_make.fit_transform(test_ml['HomePlanet'])
test_ml['Destination'] = lb_make.fit_transform(test_ml['Destination'])
test_ml['CryoSleep'] = lb_make.fit_transform(test_ml['CryoSleep'])
test_ml['VIP'] = lb_make.fit_transform(test_ml['VIP'])
test_ml['Transported'] = lb_make.fit_transform(test_ml['Transported'])
X_train = train_ml.set_index(['PassengerId', 'Name'])
y_x = cabin_train.Cabin_x.ravel()
y_z = cabin_train.Cabin_z.ravel()
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = test_ml.set_index(['PassengerId', 'Name'])
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())