import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df
train_df.isnull().sum()
train_df.describe()
train_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].boxplot(figsize=(14, 7), vert=False)
train_df[train_df['FoodCourt'] > 6500.0]
xported_vals = train_df['Transported'].value_counts()
vip_vals = train_df['VIP'].value_counts()
cryo_vals = train_df['CryoSleep'].value_counts()
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
plot_home_dest(train_df)
train_df['Age'].plot.hist()

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
plot_passenger_spending(train_df)
transported_df = train_df[train_df['Transported']]
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
sleepers_df = train_df[train_df['CryoSleep'] == True]
sleepers_df.describe()
sleepers_df.isnull().sum()
sleepers_df['Transported'].value_counts().plot.pie()
plot_home_dest(sleepers_df)
train_df[train_df['CryoSleep'].isnull()]
train_df[train_df['RoomService'].isnull()]
train_df[train_df['FoodCourt'].isnull()]
train_df[train_df['ShoppingMall'].isnull()]
train_df[train_df['Spa'].isnull()]
train_df[train_df['VRDeck'].isnull()]
train_df['Cabin'] = train_df['Cabin'].fillna('U/-1/U')
train_df[['Deck', 'CabinNum', 'ShipSide']] = train_df['Cabin'].str.split('/', expand=True)
train_df['CabinNum'] = train_df['CabinNum'].astype(int)
cabin_occ = train_df.groupby(['Cabin'])['Cabin'].transform('count')
train_df['CabinOcc'] = cabin_occ
train_df['Deck'].value_counts()
train_df[['Deck', 'HomePlanet', 'ShipSide']].groupby(['Deck', 'HomePlanet']).count()
train_df[['HomePlanet', 'ShipSide', 'Deck']].groupby(['HomePlanet', 'ShipSide']).count()
train_df[['Deck', 'Destination', 'ShipSide']].groupby(['Deck', 'Destination']).count()
train_df.loc[train_df['HomePlanet'].isna() & train_df['Deck'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
train_df.loc[train_df['HomePlanet'].isna() & train_df['Deck'].isin(['G']), 'HomePlanet'] = 'Earth'
train_df.loc[train_df['HomePlanet'].isna()]
train_df[['PassGroup', 'PassNum']] = train_df['PassengerId'].str.split('_', expand=True)
train_df['PassGroup'] = train_df['PassGroup'].astype(int)
train_df['PassNum'] = train_df['PassNum'].astype(int)
group_occ = train_df.groupby(['PassGroup'])['PassGroup'].transform('count')
train_df['GroupOcc'] = group_occ

def plot_deck_and_side(df):
    deck_vals = df['Deck'].value_counts().sort_index()
    side_vals = df['ShipSide'].value_counts()
    (fig, axes) = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)
    axes[0].pie(deck_vals, labels=deck_vals.index, startangle=90, autopct='%.1f%%')
    axes[0].set_title('Deck')
    axes[1].pie(side_vals, labels=side_vals.index, startangle=90, autopct='%.1f%%')
    axes[1].set_title('Side')
plot_deck_and_side(train_df)
transported_df = train_df[train_df['Transported']]
plot_deck_and_side(transported_df)
train_df.isnull().sum()
train_df['Age'] = train_df['Age'].fillna(train_df.groupby('HomePlanet')['Age'].transform('median'))
train_df['Age'] = train_df['Age'].fillna(27.0)
train_df['Name'] = train_df['Name'].fillna('Name Unknown')
train_df[['FName', 'LName']] = train_df['Name'].str.split(' ', expand=True)
train_df['HomePlanet'] = train_df['HomePlanet'].fillna('Unknown')
train_df['Destination'] = train_df['Destination'].fillna('TRAPPIST-1e')
train_df['VIP'] = train_df['VIP'].fillna(False)
train_df['RoomService'] = train_df['RoomService'].fillna(0.0)
train_df['FoodCourt'] = train_df['FoodCourt'].fillna(0.0)
train_df['ShoppingMall'] = train_df['ShoppingMall'].fillna(0.0)
train_df['Spa'] = train_df['Spa'].fillna(0.0)
train_df['VRDeck'] = train_df['VRDeck'].fillna(0.0)
train_df['FoodSpend'] = train_df['RoomService'] + train_df['FoodCourt']
train_df['TotalSpend'] = train_df['FoodSpend'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck']
train_df['PctRoomService'] = train_df['RoomService'] / train_df['TotalSpend']
train_df['PctFoodCourt'] = train_df['FoodCourt'] / train_df['TotalSpend']
train_df['PctFoodSpend'] = train_df['FoodSpend'] / train_df['TotalSpend']
train_df['PctShoppingMall'] = train_df['ShoppingMall'] / train_df['TotalSpend']
train_df['PctSpa'] = train_df['Spa'] / train_df['TotalSpend']
train_df['PctVRDeck'] = train_df['VRDeck'] / train_df['TotalSpend']
pct_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck', 'PctFoodSpend']
train_df[pct_cols] = train_df[pct_cols].fillna(0.0)
train_df['CryoSleep'] = train_df['CryoSleep'].fillna(train_df['TotalSpend'] == 0.0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = DecisionTreeClassifier(max_depth=5)