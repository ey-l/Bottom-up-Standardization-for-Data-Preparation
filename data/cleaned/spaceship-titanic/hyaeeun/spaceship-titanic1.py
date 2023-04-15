import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(df.shape)
df.head()
df['Transported'].value_counts()
df.info()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 3))
sns.histplot(df['RoomService'], bins=100)


print('\n')

print('\n')

print('\n')

df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, annot=True)

import matplotlib.pyplot as plt
import seaborn as sns
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (index, column) in enumerate(num_cols):
    print('index:', index)
    sns.barplot(x='HomePlanet', y=column, data=df, ax=axs[index])
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=df, ax=axs[idx1][idx2])
df.groupby('CryoSleep')['FoodCourt'].mean()
df['Cabin']

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
df['Age_cat'] = df['Age'].apply(lambda x: get_category(x))
plt.figure(figsize=(10, 6))
sns.barplot(x='Age_cat', y='Transported', data=df)
df.drop('Age_cat', axis=1, inplace=True)
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.shape
print(df[df['RoomService'] > 8000].shape[0])
df[df['RoomService'] > 8000]
corr_df = df.corr()
sns.heatmap(corr_df, annot=True)

cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(cols), figsize=(30, 4))
for (index, col) in enumerate(cols):
    sns.scatterplot(x=col, y='Transported', data=df, ax=axs[index])
print(df[df['RoomService'] > 8000].shape[0])
print(df[df['FoodCourt'] > 15000].shape[0])
print(df[df['ShoppingMall'] > 10000].shape[0])
print(df[df['Spa'] > 12000].shape[0])
print(df[df['VRDeck'] > 14000].shape[0])
print(df_test[df_test['RoomService'] > 8000].shape[0])
print(df_test[df_test['FoodCourt'] > 15000].shape[0])
print(df_test[df_test['ShoppingMall'] > 10000].shape[0])
print(df_test[df_test['Spa'] > 12000].shape[0])
print(df_test[df_test['VRDeck'] > 14000].shape[0])

def outlier_remove(df):
    outlier_r = df[df['RoomService'] > 8000].index
    df = df.drop(outlier_r, axis=0)
    outlier_f = df[df['FoodCourt'] > 15000].index
    df = df.drop(outlier_f, axis=0)
    outlier_sm = df[df['ShoppingMall'] > 9000].index
    df = df.drop(outlier_sm, axis=0)
    outlier_s = df[df['Spa'] > 12000].index
    df = df.drop(outlier_s, axis=0)
    outlier_v = df[df['VRDeck'] > 14000].index
    df = df.drop(outlier_v, axis=0)
    return df
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 3))
sns.histplot(df['RoomService'], bins=100)

cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(cols), figsize=(30, 4))
for (index, col) in enumerate(cols):
    sns.scatterplot(x=col, y='Transported', data=df, ax=axs[index])
corr_df = df.corr()
sns.heatmap(corr_df, annot=True)


def needXcol_remove(df):
    df = df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
    return df

def cat_fill(df):
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0])
    return df

def num_fill(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())
    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].median())
    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].median())
    df['Spa'] = df['Spa'].fillna(df['Spa'].median())
    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].median())
    return df
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

def label_encoding(df):
    le = LabelEncoder()
    df['CryoSleep'] = le.fit_transform(df['CryoSleep'])
    df['VIP'] = le.fit_transform(df['VIP'])
    return df

def onehot_encoding(df):
    onehot_cols = ['HomePlanet', 'Destination']
    df_oh = pd.get_dummies(df[onehot_cols], drop_first=True)
    df = pd.concat([df, df_oh], axis=1)
    df = df.drop(onehot_cols, axis=1)
    return df
from sklearn.preprocessing import StandardScaler

def scaling(df):
    scale_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    scaler = StandardScaler()