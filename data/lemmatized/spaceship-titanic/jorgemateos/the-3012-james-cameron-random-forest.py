import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input0.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.sample(10)
_input0.sample(10)
_input1 = _input1.set_index('PassengerId', inplace=False)
_input0 = _input0.set_index('PassengerId', inplace=False)
_input1['HomePlanet'].unique()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode(), inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode(), inplace=False)
pd.get_dummies(_input1['HomePlanet']).sum().plot.bar(rot=0, color='darkorange', figsize=(15, 7))
sns.countplot(data=_input1, x='HomePlanet', hue='Transported')
_input1 = pd.concat([_input1, pd.get_dummies(_input1['HomePlanet'])], axis=1).drop(columns={'HomePlanet'})
_input0 = pd.concat([_input0, pd.get_dummies(_input0['HomePlanet'])], axis=1).drop(columns={'HomePlanet'})
pd.get_dummies(_input1['CryoSleep']).sum().plot.bar(rot=0, color='skyblue', figsize=(15, 7))
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
_input1 = pd.concat([_input1, pd.get_dummies(_input1['CryoSleep'])], axis=1).rename(columns={True: 'cryosleep', False: 'no_cryosleeper'}).drop(columns={'CryoSleep', 'no_cryosleeper'})
_input0 = pd.concat([_input0, pd.get_dummies(_input0['CryoSleep'])], axis=1).rename(columns={True: 'cryosleep', False: 'no_cryosleeper'}).drop(columns={'CryoSleep', 'no_cryosleeper'})
_input1['Cabin'] = _input1['Cabin'].fillna('x/0/x', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('x/0/x', inplace=False)
_input1['Side'] = [_input1['Cabin'].str.split('/')[x][2] for x in range(len(_input1))]
_input0['Side'] = [_input0['Cabin'].str.split('/')[x][2] for x in range(len(_input0))]
pd.get_dummies(_input1['Side']).sum().plot.bar(rot=0, color='darkgrey', figsize=(15, 7))
sns.countplot(data=_input1, x='Side', hue='Transported')
_input1 = pd.concat([_input1, pd.get_dummies(_input1['Side'])], axis=1).drop(columns={'Side', 'x', 'Cabin', 'P'}).rename(columns={'S': 'starboard_cabin'})
_input0 = pd.concat([_input0, pd.get_dummies(_input0['Side'])], axis=1).drop(columns={'Side', 'x', 'Cabin', 'P'}).rename(columns={'S': 'starboard_cabin'})
_input1['Destination'].unique()
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode(), inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode(), inplace=False)
_input1 = pd.concat([_input1, pd.get_dummies(_input1['Destination'])], axis=1).drop(columns={'Destination'}).rename(columns={'TRAPPIST-1e': 'destination_trappist', 'PSO J318.5-22': 'destination_pso', '55 Cancri e': 'destination_cancri'})
_input0 = pd.concat([_input0, pd.get_dummies(_input0['Destination'])], axis=1).drop(columns={'Destination'}).rename(columns={'TRAPPIST-1e': 'destination_trappist', 'PSO J318.5-22': 'destination_pso', '55 Cancri e': 'destination_cancri'})
_input1['Age'].plot.hist(bins=20, color='lightgreen', figsize=(15, 7))
_input1['Age'] = _input1['Age'].fillna(np.random.normal(_input1['Age'].mean(), _input1['Age'].std()), inplace=False)
_input0['Age'] = _input0['Age'].fillna(np.random.normal(_input1['Age'].mean(), _input1['Age'].std()), inplace=False)
_input1['Age'].plot.hist(bins=20, color='darkgreen', figsize=(15, 7))
scaler = MinMaxScaler()
_input1['norm_age'] = scaler.fit_transform(np.array(_input1['Age']).reshape(-1, 1))
_input1 = _input1.drop(columns={'Age'}, inplace=False)
_input0['norm_age'] = scaler.fit_transform(np.array(_input0['Age']).reshape(-1, 1))
_input0 = _input0.drop(columns={'Age'}, inplace=False)
pd.get_dummies(_input1['VIP']).sum().plot.bar(color='silver', figsize=(15, 7))
_input1 = pd.concat([_input1, pd.get_dummies(_input1['VIP'])], axis=1).drop(columns={'VIP', False}).rename(columns={True: 'VIP'})
_input0 = pd.concat([_input0, pd.get_dummies(_input0['VIP'])], axis=1).drop(columns={'VIP', False}).rename(columns={True: 'VIP'})
_input1['luxury_expenses'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['luxury_expenses'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1 = _input1.drop(columns={'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'}, inplace=False)
_input0 = _input0.drop(columns={'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'}, inplace=False)
_input1['luxury_expenses'] = _input1['luxury_expenses'].mask((_input1['VIP'] == 1) & _input1['luxury_expenses'], _input1['luxury_expenses'].loc[_input1['VIP'] == 1].mean(), inplace=False)
_input1['luxury_expenses'] = _input1['luxury_expenses'].fillna(0.0, inplace=False)
_input0['luxury_expenses'] = _input0['luxury_expenses'].mask((_input1['VIP'] == 1) & _input1['luxury_expenses'], _input1['luxury_expenses'].loc[_input1['VIP'] == 1].mean(), inplace=False)
_input0['luxury_expenses'] = _input0['luxury_expenses'].fillna(0.0, inplace=False)
_input1['norm_luxury_expenses'] = scaler.fit_transform(np.array(_input1['luxury_expenses']).reshape(-1, 1))
_input0['norm_luxury_expenses'] = scaler.fit_transform(np.array(_input0['luxury_expenses']).reshape(-1, 1))
_input1 = _input1.drop(columns={'luxury_expenses'}, inplace=False)
_input0 = _input0.drop(columns={'luxury_expenses'}, inplace=False)
_input1 = pd.concat([_input1, pd.get_dummies(_input1['Transported'])], axis=1).drop(columns={'Transported', False}).rename(columns={True: 'transported'})
_input1.head()
_input0.head()
df_X = _input1.drop('transported', axis=1)
df_y = _input1['transported']
random_forest = RandomForestClassifier(max_depth=10, random_state=132)
(X_train, X_test, y_train, y_test) = train_test_split(df_X, df_y, test_size=0.3, random_state=0)