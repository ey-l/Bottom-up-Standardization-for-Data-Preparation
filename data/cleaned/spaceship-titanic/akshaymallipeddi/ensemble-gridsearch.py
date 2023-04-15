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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df.head
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df.head
train_df.columns

def get_group(text):
    group = int(text.split('_')[0])
    return group
train_df['Group'] = train_df['PassengerId'].apply(lambda x: get_group(x))
test_df['Group'] = test_df['PassengerId'].apply(lambda x: get_group(x))
print(train_df['Group'].nunique())
print(train_df['Group'])
print(train_df['Group'].value_counts(sort=False)[train_df['Group']])
print(type(train_df['Group'].value_counts(sort=False)[train_df['Group']]))
train_df['Group_mem_count'] = train_df['Group'].value_counts(sort=False)[train_df['Group']].tolist()
test_df['Group_mem_count'] = test_df['Group'].value_counts(sort=False)[test_df['Group']].tolist()
print(train_df.columns)
print(test_df.columns)
print(train_df['HomePlanet'].unique())
print(test_df['HomePlanet'].unique())
print(train_df['Destination'].unique())
print(test_df['Destination'].unique())
train_df['Destination'] = train_df['Destination'].fillna(value=train_df['Destination'].mode().iloc[0])
test_df['Destination'] = test_df['Destination'].fillna(value=test_df['Destination'].mode().iloc[0])
print(train_df['Destination'].isnull().sum())
print(test_df['Destination'].isnull().sum())
train_df['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=True)
test_df['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], [0, 1, 2], inplace=True)
train_labels = train_df.Transported.values.astype(int)
len(train_labels)
for col in train_df:
    print(col)
    print(train_df[col].mode().iloc[0])
    print(type(train_df[col].mode()))
    print(train_df[col].mode().iloc[[0]])
    print('Checking for null')
    print(train_df[col].isnull().sum())
    print('---' * 10)
train_df['HomePlanet'] = train_df['HomePlanet'].fillna(value=train_df['HomePlanet'].mode().iloc[0])
test_df['HomePlanet'] = test_df['HomePlanet'].fillna(value=test_df['HomePlanet'].mode().iloc[0])
print(train_df['HomePlanet'].nunique())
train_df['HomePlanet'].replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=True)
test_df['HomePlanet'].replace(['Europa', 'Earth', 'Mars'], [0, 1, 2], inplace=True)
print(train_df['HomePlanet'].mode())
print(test_df['HomePlanet'].mode())
print(train_df['HomePlanet'].unique())
print(test_df['HomePlanet'].unique())
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df[cols] = train_df[cols].fillna(value=train_df[cols].mean())
test_df[cols] = test_df[cols].fillna(value=test_df[cols].mean())
train_df[cols].mean()
train_df['CryoSleep'] = train_df['CryoSleep'].fillna(value=train_df['CryoSleep'].mode().iloc[0])
test_df['CryoSleep'] = test_df['CryoSleep'].fillna(value=test_df['CryoSleep'].mode().iloc[0])
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)
train_df['Age'] = train_df['Age'].fillna(value=train_df['Age'].mean())
test_df['Age'] = test_df['Age'].fillna(value=test_df['Age'].mean())
train_df['Total_spent'] = train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck']
test_df['Total_spent'] = test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck']
print(train_df.columns)
required_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'Total_spent', 'Group_mem_count', 'ShoppingMall', 'Spa', 'VRDeck', 'FoodCourt', 'RoomService']
print(train_df['Total_spent'].isnull().sum())
print(test_df['Total_spent'].isnull().sum())
final_train_df = train_df[required_cols]
final_test_df = test_df[required_cols]
(x_train, x_val, y_train, y_val) = train_test_split(final_train_df, train_labels, test_size=0.1, shuffle=True)
print(x_train.shape)
print(x_val.shape)

def train_randomforest(x_train, y_train, x_val, y_val):
    classifier = RandomForestClassifier(n_estimators=1000)