import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df = pd.DataFrame(_input1)
train_df
train_df = train_df.interpolate(method='linear', limit_direction='forward', inplace=False)
train_df = train_df.fillna(method='bfill', inplace=False)
train_df.isnull().sum()
train_df[['DeckName', 'DeckNumber', 'DeckSide']] = train_df['Cabin'].str.split('/', expand=True)
train_df[['Group', 'ID']] = train_df['PassengerId'].str.split('_', expand=True)
train_df['HomePlanet'] = train_df['HomePlanet'].map({'Europa': 1, 'Earth': 2, 'Mars': 3}).astype('int8')
train_df['DeckSide'] = train_df['DeckSide'].map({'P': 1, 'S': 2}).astype('int8')
train_df['DeckName'] = train_df['DeckName'].map({'B': 1, 'F': 2, 'A': 3, 'G': 4, 'E': 5, 'D': 6, 'C': 7, 'T': 8}).astype('int8')
train_df['Destination'] = train_df['Destination'].map({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}).astype('int8')
train_df
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)
train_df['VIP'] = train_df['VIP'].astype(int)
train_df['Transported'] = train_df['Transported'].astype(int)
train_df.head(100)
x_train = train_df[['Group', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckName', 'DeckNumber', 'DeckSide']]
y_train = train_df['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.33, random_state=52)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=80, max_depth=19)