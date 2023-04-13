import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input1.columns
_input1.isna().sum()
input = _input1.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'], axis=1)
target = _input1['Transported']
input
input['Destination'].unique()
from sklearn.preprocessing import LabelEncoder
le_destination = LabelEncoder()
le_vip = LabelEncoder()
input['Destination_n'] = le_destination.fit_transform(input['Destination'])
input['VIP_n'] = le_vip.fit_transform(input['VIP'])
input
input.isna().sum()
target.isna().sum()
input['CryoSleep'] = input['CryoSleep'].fillna(input['CryoSleep'].value_counts().index[0], inplace=False)
input['Destination'] = input['Destination'].fillna(input['Destination'].value_counts().index[0], inplace=False)
input['VIP'] = input['VIP'].fillna(input['VIP'].value_counts().index[0], inplace=False)
input['Age'] = input['Age'].fillna(input['Age'].mean(), inplace=False)
input = input.drop(['Destination', 'VIP'], axis=1, inplace=False)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input0 = _input0.drop(['HomePlanet', 'Cabin', 'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
_input0['Destination_n'] = le_destination.fit_transform(_input0['Destination'])
_input0['VIP_n'] = le_vip.fit_transform(_input0['VIP'])
output = _input0.drop(['PassengerId'], axis=1)
output.head()
output['CryoSleep'] = output['CryoSleep'].fillna(output['CryoSleep'].value_counts().index[0], inplace=False)
output['Destination_n'] = output['Destination_n'].fillna(output['Destination'].value_counts().index[0], inplace=False)
output['VIP_n'] = output['VIP_n'].fillna(output['VIP'].value_counts().index[0], inplace=False)
output['Age'] = output['Age'].fillna(output['Age'].mean(), inplace=False)
output = output.drop(['Destination', 'VIP'], axis=1, inplace=False)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(input, target, test_size=0.1, random_state=25)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=25, criterion='entropy', max_features='log2')