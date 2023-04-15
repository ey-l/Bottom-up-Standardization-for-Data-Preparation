import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
import pandas_profiling
pandas_profiling.ProfileReport(train)
train.info()
train.isnull().sum()
train.interpolate(inplace=True)
train.fillna(method='bfill', inplace=True)
train.head()
train.nunique()
train['Destination'].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['HomePlanet'] = le.fit_transform(train['HomePlanet'])
train['Destination'] = le.fit_transform(train['Destination'])
train[['DeckName', 'DeckNumber', 'DeckSide']] = train['Cabin'].str.split('/', expand=True)
train[['Group', 'ID']] = train['PassengerId'].str.split('_', expand=True)
train.head()
train['DeckName'] = le.fit_transform(train['DeckName'])
train['DeckSide'] = le.fit_transform(train['DeckSide'])
train.head()
train['CryoSleep'] = train['CryoSleep'].astype(int)
train['VIP'] = train['VIP'].astype(int)
train['Transported'] = train['Transported'].astype(int)
train['Total_Exp'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
train.head()
x_train = train[['Group', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckName', 'DeckNumber', 'DeckSide', 'Total_Exp']]
y_train = train['Transported']
pandas_profiling.ProfileReport(x_train)
x_train.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()