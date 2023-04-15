import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import set_config

plt.figure(figsize=(20, 15))
set_config(display='diagram')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')
spaceship = pd.read_csv('data/input/spaceship-titanic/train.csv')
spaceship_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
(spaceship.shape, spaceship_test.shape)
spaceship_train = spaceship.copy()
spaceship.sample(10)
spaceship.info()
plt.figure(figsize=(22, 4))
sns.heatmap(spaceship.drop(columns='Transported').isna().sum().to_frame(name='train_na').T, cmap='Spectral', annot=True, fmt='0.0f').set_title('count missing values', fontsize=14)
spaceship.describe(include='all')

def SplitGroupCabin(df):
    df['passengerGroup'] = df.PassengerId.str.split('_').str[0]
    df['cabinDeck'] = df.Cabin.str.split('/').str[0]
    df['cabinNum'] = df.Cabin.str.split('/').str[1]
    df['cabinSide'] = df.Cabin.str.split('/').str[2]
    df.drop(columns=['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
    return df
SplitGroupCabin(spaceship)
spaceship.head(5)
spaceship.info()
plt.figure(figsize=(22, 4))
sns.heatmap(spaceship.drop(columns='Transported').isna().sum().to_frame(name='train_na').T, cmap='Spectral', annot=True, fmt='0.0f').set_title('count missing values', fontsize=14)
cat_col = []
num_col = []

def convert_types(df):
    global cat_col, num_col
    cat_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabinDeck', 'cabinSide']
    num_col = ['Age', 'passengerGroup', 'cabinNum', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for i in cat_col:
        df[i] = df[i].astype('category')
    for i in num_col:
        df[i] = df[i].astype('float')
    return df
convert_types(spaceship)
spaceship.head(5)
(fig, axes) = plt.subplots(6, 2, figsize=(20, 35))
idx = 0
for col in cat_col:
    sns.countplot(data=spaceship, y=col, palette='magma', orient='h', ax=axes[idx][0]).set_title(f'Count of {col}', fontsize='16')
    sns.countplot(data=spaceship, y=col, palette='mako', orient='h', hue='Transported', ax=axes[idx][1]).set_title(f'Count of {col} per transported', fontsize='16')
    idx += 1

(fig, axes) = plt.subplots(8, 2, figsize=(20, 45))
idx = 0
for col in num_col:
    sns.kdeplot(data=spaceship, x=col, palette='Greens', fill=True, hue='Transported', ax=axes[idx][0]).set_title(f'Distribution of {col}', fontsize='16')
    sns.boxplot(data=spaceship, x=col, palette='flare', y='Transported', orient='h', ax=axes[idx][1]).set_title(f'BoxPlot of {col}', fontsize='16')
    idx += 1

from sklearn.impute import SimpleImputer
impute_cols1 = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabinDeck', 'cabinSide']
impute_cols2 = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
impute_cols3 = ['Age', 'cabinNum']