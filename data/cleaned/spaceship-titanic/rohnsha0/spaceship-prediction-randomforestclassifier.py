import numpy as np
import pandas as pd
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.shape
df.columns
df.isna().sum()
input = df.drop(['PassengerId', 'HomePlanet', 'Cabin', 'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'], axis=1)
target = df['Transported']
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
input['CryoSleep'].fillna(input['CryoSleep'].value_counts().index[0], inplace=True)
input['Destination'].fillna(input['Destination'].value_counts().index[0], inplace=True)
input['VIP'].fillna(input['VIP'].value_counts().index[0], inplace=True)
input['Age'].fillna(input['Age'].mean(), inplace=True)
input.drop(['Destination', 'VIP'], axis=1, inplace=True)
df2 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df2
df2 = df2.drop(['HomePlanet', 'Cabin', 'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
df2['Destination_n'] = le_destination.fit_transform(df2['Destination'])
df2['VIP_n'] = le_vip.fit_transform(df2['VIP'])
output = df2.drop(['PassengerId'], axis=1)
output.head()
output['CryoSleep'].fillna(output['CryoSleep'].value_counts().index[0], inplace=True)
output['Destination_n'].fillna(output['Destination'].value_counts().index[0], inplace=True)
output['VIP_n'].fillna(output['VIP'].value_counts().index[0], inplace=True)
output['Age'].fillna(output['Age'].mean(), inplace=True)
output.drop(['Destination', 'VIP'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(input, target, test_size=0.1, random_state=25)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=25, criterion='entropy', max_features='log2')