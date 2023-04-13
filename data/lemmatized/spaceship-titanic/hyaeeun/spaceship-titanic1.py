import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1.shape)
_input1.head()
_input1['Transported'].value_counts()
_input1.info()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 3))
sns.histplot(_input1['RoomService'], bins=100)
print('\n')
print('\n')
print('\n')
_input1.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
corr = _input1.corr()
sns.heatmap(corr, annot=True)
import matplotlib.pyplot as plt
import seaborn as sns
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (index, column) in enumerate(num_cols):
    print('index:', index)
    sns.barplot(x='HomePlanet', y=column, data=_input1, ax=axs[index])
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=_input1, ax=axs[idx1][idx2])
_input1.groupby('CryoSleep')['FoodCourt'].mean()
_input1['Cabin']

def get_category(age):
    cat = ''
    if age <= 15:
        cat = '0~15'
    elif age <= 25:
        cat = '16~25'
    elif age <= 35:
        cat = '26~35'
    elif age <= 45:
        cat = '36~45'
    elif age <= 55:
        cat = '46~55'
    elif age <= 65:
        cat = '56~65'
    else:
        cat = '61~'
    return cat
_input1['Age_cat'] = _input1['Age'].apply(lambda x: get_category(x))
plt.figure(figsize=(10, 6))
sns.barplot(x='Age_cat', y='Transported', data=_input1)
_input1 = _input1.drop('Age_cat', axis=1, inplace=False)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.shape
print(_input1[_input1['RoomService'] > 8000].shape[0])
_input1[_input1['RoomService'] > 8000]
corr_df = _input1.corr()
sns.heatmap(corr_df, annot=True)
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(cols), figsize=(30, 4))
for (index, col) in enumerate(cols):
    sns.scatterplot(x=col, y='Transported', data=_input1, ax=axs[index])
print(_input1[_input1['RoomService'] > 8000].shape[0])
print(_input1[_input1['FoodCourt'] > 15000].shape[0])
print(_input1[_input1['ShoppingMall'] > 10000].shape[0])
print(_input1[_input1['Spa'] > 12000].shape[0])
print(_input1[_input1['VRDeck'] > 14000].shape[0])
print(_input0[_input0['RoomService'] > 8000].shape[0])
print(_input0[_input0['FoodCourt'] > 15000].shape[0])
print(_input0[_input0['ShoppingMall'] > 10000].shape[0])
print(_input0[_input0['Spa'] > 12000].shape[0])
print(_input0[_input0['VRDeck'] > 14000].shape[0])

def outlier_remove(df):
    outlier_r = _input1[_input1['RoomService'] > 8000].index
    _input1 = _input1.drop(outlier_r, axis=0)
    outlier_f = _input1[_input1['FoodCourt'] > 15000].index
    _input1 = _input1.drop(outlier_f, axis=0)
    outlier_sm = _input1[_input1['ShoppingMall'] > 9000].index
    _input1 = _input1.drop(outlier_sm, axis=0)
    outlier_s = _input1[_input1['Spa'] > 12000].index
    _input1 = _input1.drop(outlier_s, axis=0)
    outlier_v = _input1[_input1['VRDeck'] > 14000].index
    _input1 = _input1.drop(outlier_v, axis=0)
    return _input1
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 3))
sns.histplot(_input1['RoomService'], bins=100)
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(cols), figsize=(30, 4))
for (index, col) in enumerate(cols):
    sns.scatterplot(x=col, y='Transported', data=_input1, ax=axs[index])
corr_df = _input1.corr()
sns.heatmap(corr_df, annot=True)

def needXcol_remove(df):
    _input1 = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
    return _input1

def cat_fill(df):
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0])
    _input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0])
    _input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0])
    _input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0])
    return _input1

def num_fill(df):
    _input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
    _input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median())
    _input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median())
    _input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median())
    _input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median())
    _input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median())
    return _input1
_input1.isnull().sum()
from sklearn.preprocessing import LabelEncoder

def label_encoding(df):
    le = LabelEncoder()
    _input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
    _input1['VIP'] = le.fit_transform(_input1['VIP'])
    return _input1

def onehot_encoding(df):
    onehot_cols = ['HomePlanet', 'Destination']
    df_oh = pd.get_dummies(_input1[onehot_cols], drop_first=True)
    _input1 = pd.concat([_input1, df_oh], axis=1)
    _input1 = _input1.drop(onehot_cols, axis=1)
    return _input1
from sklearn.preprocessing import StandardScaler

def scaling(df):
    scale_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    scaler = StandardScaler()