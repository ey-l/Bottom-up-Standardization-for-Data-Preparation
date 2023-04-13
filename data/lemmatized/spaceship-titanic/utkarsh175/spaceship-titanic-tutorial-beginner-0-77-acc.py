import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['HomePlanet'] = le.fit_transform(_input1['HomePlanet'])
_input0['HomePlanet'] = le.fit_transform(_input0['HomePlanet'])
_input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
_input0['CryoSleep'] = le.fit_transform(_input0['CryoSleep'])
_input1['Cabin'] = le.fit_transform(_input1['Cabin'])
_input0['Cabin'] = le.fit_transform(_input0['Cabin'])
_input1['Destination'] = le.fit_transform(_input1['Destination'])
_input0['Destination'] = le.fit_transform(_input0['Destination'])
_input1.info()
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
_input1.info()
_input0.info()
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].median(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].median(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].median(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].median(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].median(), inplace=False)
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input1['VIP'] = le.fit_transform(_input1['VIP'])
_input0['VIP'] = le.fit_transform(_input0['VIP'])
_input1['Transported'] = le.fit_transform(_input1['Transported'])
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
_input1.info()
_input0.info()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
from sklearn.ensemble import RandomForestClassifier
X_train = _input1.drop('Transported', axis=1)
y_train = _input1[['Transported']]
X_test = _input0
y_train['Transported']
X_train.info()
rf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=7)