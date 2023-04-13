import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
from sklearn.impute import KNNImputer
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(method='bfill', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(method='bfill', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(method='bfill', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(method='bfill', inplace=False)
_input1['Age'] = _input1['Age'].fillna(value=_input1['Age'].mode()[0], inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(method='bfill', inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(method='bfill', inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(method='bfill', inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(method='bfill', inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(method='bfill', inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(method='bfill', inplace=False)
_input1['Name'] = _input1['Name'].fillna('null', inplace=False)
print(_input1.isnull().sum())
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
group = []
group_member = []
for i in range(_input1['PassengerId'].size):
    str_left = _input1['PassengerId'][i].split('_')[0]
    group.append(str_left)
    str_right = _input1['PassengerId'][i].split('_')[1]
    group_member.append(str_right)
group = pd.DataFrame(group, columns=['group'], dtype=np.float)
group_member = pd.DataFrame(group_member, columns=['group_member'], dtype=np.float)