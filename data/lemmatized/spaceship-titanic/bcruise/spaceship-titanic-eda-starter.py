import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.isnull().sum()
_input1.describe()
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].boxplot(figsize=(14, 7), vert=False)
_input1[_input1['FoodCourt'] > 6500.0]
xported_vals = _input1['Transported'].value_counts()
vip_vals = _input1['VIP'].value_counts()
cryo_vals = _input1['CryoSleep'].value_counts()
(fig, axes) = plt.subplots(1, 3)
fig.set_size_inches(12, 4)
axes[0].pie(xported_vals, labels=xported_vals.index, startangle=90, autopct='%.1f%%')
axes[0].set_title('Transported')
axes[1].pie(vip_vals, labels=vip_vals.index, startangle=90, autopct='%.1f%%')
axes[1].set_title('VIP')
axes[2].pie(cryo_vals, labels=cryo_vals.index, startangle=90, autopct='%.1f%%')
axes[2].set_title('Cryo Sleep')

def plot_home_dest(df):
    home_vals = df['HomePlanet'].value_counts()
    dest_vals = df['Destination'].value_counts()
    (fig, axes) = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)
    axes[0].pie(home_vals, labels=home_vals.index, startangle=90, autopct='%.1f%%')
    axes[0].set_title('Home Planet')
    axes[1].pie(dest_vals, labels=dest_vals.index, startangle=90, autopct='%.1f%%')
    axes[1].set_title('Destination')
plot_home_dest(_input1)
_input1['Age'].plot.hist()

def plot_passenger_spending(df):
    (fig, axes) = plt.subplots(2, 3)
    fig.set_size_inches(12, 8)
    axes[0, 0].hist(df['RoomService'], bins=20)
    axes[0, 0].set_title('Room Service')
    axes[0, 1].hist(df['FoodCourt'], bins=20)
    axes[0, 1].set_title('Food Court')
    axes[1, 0].hist(df['ShoppingMall'], bins=20)
    axes[1, 0].set_title('Shopping Mall')
    axes[1, 1].hist(df['Spa'], bins=20)
    axes[1, 1].set_title('Spa')
    axes[1, 2].hist(df['VRDeck'], bins=20)
    axes[1, 2].set_title('VR Deck')
plot_passenger_spending(_input1)
transported_df = _input1[_input1['Transported']]
transported_df
vip_vals = transported_df['VIP'].value_counts()
cryo_vals = transported_df['CryoSleep'].value_counts()
(fig, axes) = plt.subplots(1, 2)
fig.set_size_inches(9, 4)
axes[0].pie(vip_vals, labels=vip_vals.index, startangle=90, autopct='%.1f%%')
axes[0].set_title('VIP')
axes[1].pie(cryo_vals, labels=cryo_vals.index, startangle=90, autopct='%.1f%%')
axes[1].set_title('Cryo Sleep')
plot_home_dest(transported_df)
transported_df['Age'].plot.hist()
plot_passenger_spending(transported_df)
sleepers_df = _input1[_input1['CryoSleep'] == True]
sleepers_df.describe()
sleepers_df.isnull().sum()
sleepers_df['Transported'].value_counts().plot.pie()
plot_home_dest(sleepers_df)
_input1[_input1['CryoSleep'].isnull()]
_input1[_input1['RoomService'].isnull()]
_input1[_input1['FoodCourt'].isnull()]
_input1[_input1['ShoppingMall'].isnull()]
_input1[_input1['Spa'].isnull()]
_input1[_input1['VRDeck'].isnull()]
_input1['Cabin'] = _input1['Cabin'].fillna('U/-1/U')
_input1[['Deck', 'CabinNum', 'ShipSide']] = _input1['Cabin'].str.split('/', expand=True)
_input1['CabinNum'] = _input1['CabinNum'].astype(int)
cabin_occ = _input1.groupby(['Cabin'])['Cabin'].transform('count')
_input1['CabinOcc'] = cabin_occ
_input1['Deck'].value_counts()
_input1[['Deck', 'HomePlanet', 'ShipSide']].groupby(['Deck', 'HomePlanet']).count()
_input1[['HomePlanet', 'ShipSide', 'Deck']].groupby(['HomePlanet', 'ShipSide']).count()
_input1[['Deck', 'Destination', 'ShipSide']].groupby(['Deck', 'Destination']).count()
_input1.loc[_input1['HomePlanet'].isna() & _input1['Deck'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
_input1.loc[_input1['HomePlanet'].isna() & _input1['Deck'].isin(['G']), 'HomePlanet'] = 'Earth'
_input1.loc[_input1['HomePlanet'].isna()]
_input1[['PassGroup', 'PassNum']] = _input1['PassengerId'].str.split('_', expand=True)
_input1['PassGroup'] = _input1['PassGroup'].astype(int)
_input1['PassNum'] = _input1['PassNum'].astype(int)
group_occ = _input1.groupby(['PassGroup'])['PassGroup'].transform('count')
_input1['GroupOcc'] = group_occ

def plot_deck_and_side(df):
    deck_vals = df['Deck'].value_counts().sort_index()
    side_vals = df['ShipSide'].value_counts()
    (fig, axes) = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)
    axes[0].pie(deck_vals, labels=deck_vals.index, startangle=90, autopct='%.1f%%')
    axes[0].set_title('Deck')
    axes[1].pie(side_vals, labels=side_vals.index, startangle=90, autopct='%.1f%%')
    axes[1].set_title('Side')
plot_deck_and_side(_input1)
transported_df = _input1[_input1['Transported']]
plot_deck_and_side(transported_df)
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1.groupby('HomePlanet')['Age'].transform('median'))
_input1['Age'] = _input1['Age'].fillna(27.0)
_input1['Name'] = _input1['Name'].fillna('Name Unknown')
_input1[['FName', 'LName']] = _input1['Name'].str.split(' ', expand=True)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Unknown')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input1['VIP'] = _input1['VIP'].fillna(False)
_input1['RoomService'] = _input1['RoomService'].fillna(0.0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0.0)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0.0)
_input1['Spa'] = _input1['Spa'].fillna(0.0)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0.0)
_input1['FoodSpend'] = _input1['RoomService'] + _input1['FoodCourt']
_input1['TotalSpend'] = _input1['FoodSpend'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1['PctRoomService'] = _input1['RoomService'] / _input1['TotalSpend']
_input1['PctFoodCourt'] = _input1['FoodCourt'] / _input1['TotalSpend']
_input1['PctFoodSpend'] = _input1['FoodSpend'] / _input1['TotalSpend']
_input1['PctShoppingMall'] = _input1['ShoppingMall'] / _input1['TotalSpend']
_input1['PctSpa'] = _input1['Spa'] / _input1['TotalSpend']
_input1['PctVRDeck'] = _input1['VRDeck'] / _input1['TotalSpend']
pct_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck', 'PctFoodSpend']
_input1[pct_cols] = _input1[pct_cols].fillna(0.0)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['TotalSpend'] == 0.0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = DecisionTreeClassifier(max_depth=5)