import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
import pandas_profiling
pandas_profiling.ProfileReport(_input1)
_input1.info()
_input1.isnull().sum()
_input1 = _input1.interpolate(inplace=False)
_input1 = _input1.fillna(method='bfill', inplace=False)
_input1.head()
_input1.nunique()
_input1['Destination'].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['HomePlanet'] = le.fit_transform(_input1['HomePlanet'])
_input1['Destination'] = le.fit_transform(_input1['Destination'])
_input1[['DeckName', 'DeckNumber', 'DeckSide']] = _input1['Cabin'].str.split('/', expand=True)
_input1[['Group', 'ID']] = _input1['PassengerId'].str.split('_', expand=True)
_input1.head()
_input1['DeckName'] = le.fit_transform(_input1['DeckName'])
_input1['DeckSide'] = le.fit_transform(_input1['DeckSide'])
_input1.head()
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input1['VIP'] = _input1['VIP'].astype(int)
_input1['Transported'] = _input1['Transported'].astype(int)
_input1['Total_Exp'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1.head()
x_train = _input1[['Group', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'DeckName', 'DeckNumber', 'DeckSide', 'Total_Exp']]
y_train = _input1['Transported']
pandas_profiling.ProfileReport(x_train)
x_train.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()