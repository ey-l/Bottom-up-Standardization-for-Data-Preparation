import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
bg_color = 'white'
ktcolors = ['#d0384e', '#ee6445', '#fa9b58', '#fece7c', '#fff1a8', '#f4faad', '#d1ed9c', '#97d5a4', '#5cb7aa', '#3682ba']
sns.set(rc={'font.style': 'normal', 'axes.facecolor': bg_color, 'figure.facecolor': bg_color, 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black', 'axes.grid': False, 'axes.labelsize': 20, 'figure.figsize': (5.0, 5.0), 'xtick.labelsize': 10, 'font.size': 10, 'ytick.labelsize': 10})
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.head()
df_train.info()
df_train.CryoSleep = df_train.CryoSleep.astype(bool)
df_train.VIP = df_train.VIP.astype(bool)
df_train.info()
df_test.CryoSleep = df_test.CryoSleep.astype(bool)
df_test.VIP = df_test.VIP.astype(bool)
df_train.sample(15)
df_train.drop(['Name'], axis=1, inplace=True)
df_test.drop(['Name'], axis=1, inplace=True)
df_train.HomePlanet.unique()
df_train.HomePlanet = df_train.HomePlanet.astype('category')
df_test.HomePlanet = df_test.HomePlanet.astype('category')
df_train.info()
df_train.Destination.unique()
df_train.Destination = df_train.Destination.astype('category')
df_test.Destination = df_test.Destination.astype('category')
df_train.info()
df_train.isnull().sum()
df_test.isnull().sum()
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_train.isnull().sum()
pricing_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in pricing_cols:
    df_train[col].fillna(0, inplace=True)
    df_test[col].fillna(0, inplace=True)
df_train.isnull().sum()
for col in df_train.isnull().sum().index[0:-1]:
    temp = df_train[col].value_counts().index[0]
    df_train[col] = df_train[col].fillna(temp)
    df_test[col] = df_test[col].fillna(temp)
print(f'Training NaNs:\n{df_train.isnull().sum()}\n\nTesting NaNs:\n{df_test.isnull().sum()}')
print(f'\nThe data contains {df_train.isnull().sum().sum() + df_test.isnull().sum().sum()} NaNs')
df_train = pd.get_dummies(data=df_train, columns=['HomePlanet', 'Destination'])
df_train.columns
df_test = pd.get_dummies(data=df_test, columns=['HomePlanet', 'Destination'])
df_test.columns
features = ['HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 'CryoSleep', 'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
y = df_train.Transported
X = df_train[features]
X_test = df_test[features]
model = RandomForestClassifier()