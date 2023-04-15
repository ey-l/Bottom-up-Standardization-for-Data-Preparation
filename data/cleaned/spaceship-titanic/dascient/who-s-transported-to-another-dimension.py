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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.loc[train.Cabin.isnull(), 'Cabin'] = 'Z/9999/Z'
print('All passengers in CryoSleep & VIP were transported:')
all(train.loc[(train['CryoSleep'] == True) & (train['VIP'] == True)].Transported == True)
list0 = ['F', 'G', 'E', 'B', 'D', 'A', 'T']
list1 = ['S', 'P']
for (i, v) in train.Cabin.items():
    cabin = str(v).split('/')
    try:
        train.at[i, 'Cabin_x'] = cabin[0]
    except:
        train.at[i, 'Cabin_x'] = list0[random.randint(0, 6)]
    try:
        train.at[i, 'Cabin_y'] = int(cabin[1])
    except:
        train.at[i, 'Cabin_y'] = random.randint(0, 1894)
    try:
        train.at[i, 'Cabin_z'] = cabin[2]
    except:
        train.at[i, 'Cabin_z'] = list1[random.randint(0, 1)]
for (i, v) in test.Cabin.items():
    cabin = str(v).split('/')
    try:
        test.at[i, 'Cabin_x'] = cabin[0]
    except:
        test.at[i, 'Cabin_x'] = list0[random.randint(0, 6)]
    try:
        test.at[i, 'Cabin_y'] = int(cabin[1])
    except:
        test.at[i, 'Cabin_y'] = random.randint(0, 1894)
    try:
        test.at[i, 'Cabin_z'] = cabin[2]
    except:
        test.at[i, 'Cabin_z'] = list1[random.randint(0, 1)]
train['Age'].fillna(train['Age'].median(), inplace=True)
train['VIP'].fillna(train['VIP'].median(), inplace=True)
train['RoomService'].fillna(train['RoomService'].median(), inplace=True)
train['FoodCourt'].fillna(train['FoodCourt'].median(), inplace=True)
train['ShoppingMall'].fillna(train['ShoppingMall'].median(), inplace=True)
train['Spa'].fillna(train['Spa'].median(), inplace=True)
train['VRDeck'].fillna(train['VRDeck'].median(), inplace=True)
train['Name'].fillna('John Doe', inplace=True)
train['HomePlanet'].fillna('Earth', inplace=True)
train['CryoSleep'].fillna(True, inplace=True)
train['Destination'].fillna('TRAPPIST-1e', inplace=True)
train.info()
print(len(train.loc[train.Cabin == 'Z/9999/Z']), 'null cabins (i.e. missing features)')
cabin_train = train[['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Cabin_x', 'Cabin_y', 'Cabin_z']]
cabin_test = train.loc[train.Cabin == 'Z/9999/Z'][['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
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