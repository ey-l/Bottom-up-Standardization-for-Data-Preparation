import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head
_input1.columns

def get_group(text):
    group = int(text.split('_')[0])
    return group
_input1['Group'] = _input1['PassengerId'].apply(lambda x: get_group(x))
_input0['Group'] = _input0['PassengerId'].apply(lambda x: get_group(x))
print(_input1['Group'].nunique())
print(_input1['Group'])
print(_input1['Group'].value_counts(sort=False)[_input1['Group']])
print(type(_input1['Group'].value_counts(sort=False)[_input1['Group']]))
_input1['Group_mem_count'] = _input1['Group'].value_counts(sort=False)[_input1['Group']].tolist()
_input0['Group_mem_count'] = _input0['Group'].value_counts(sort=False)[_input0['Group']].tolist()
print(_input1.columns)
print(_input0.columns)
print(_input1['HomePlanet'].unique())
print(_input0['HomePlanet'].unique())
print(_input1['Destination'].unique())
print(_input0['Destination'].unique())
_input1['Destination'] = _input1['Destination'].fillna(value=_input1['Destination'].mode().iloc[0])
_input0['Destination'] = _input0['Destination'].fillna(value=_input0['Destination'].mode().iloc[0])
print(_input1['Destination'].isnull().sum())
print(_input0['Destination'].isnull().sum())
_input1['Destination'] = _input1['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=False)
_input0['Destination'] = _input0['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=False)
train_labels = _input1.Transported.values.astype(int)
len(train_labels)
for col in _input1:
    print(col)
    print(_input1[col].mode().iloc[0])
    print(type(_input1[col].mode()))
    print(_input1[col].mode().iloc[[0]])
    print('Checking for null')
    print(_input1[col].isnull().sum())
    print('---' * 10)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(value=_input1['HomePlanet'].mode().iloc[0])
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(value=_input0['HomePlanet'].mode().iloc[0])
print(_input1['HomePlanet'].nunique())
_input1['HomePlanet'] = _input1['HomePlanet'].replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=False)
print(_input1['HomePlanet'].mode())
print(_input0['HomePlanet'].mode())
print(_input1['HomePlanet'].unique())
print(_input0['HomePlanet'].unique())
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1[cols] = _input1[cols].fillna(value=_input1[cols].mean())
_input0[cols] = _input0[cols].fillna(value=_input0[cols].mean())
_input1[cols].mean()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(value=_input1['CryoSleep'].mode().iloc[0])
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(value=_input0['CryoSleep'].mode().iloc[0])
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
_input1['Age'] = _input1['Age'].fillna(value=_input1['Age'].mean())
_input0['Age'] = _input0['Age'].fillna(value=_input0['Age'].mean())
_input1['Total_spent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['Total_spent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
print(_input1.columns)
required_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'Total_spent', 'Group_mem_count', 'ShoppingMall', 'Spa', 'VRDeck', 'FoodCourt', 'RoomService']
print(_input1['Total_spent'].isnull().sum())
print(_input0['Total_spent'].isnull().sum())
final_train_df = _input1[required_cols]
final_test_df = _input0[required_cols]
(x_train, x_val, y_train, y_val) = train_test_split(final_train_df, train_labels, test_size=0.1, shuffle=True)
print(x_train.shape)
print(x_val.shape)

def train_randomforest(x_train, y_train, x_val, y_val):
    classifier = RandomForestClassifier(n_estimators=1000)