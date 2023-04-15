import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.info()
df_test.info()
df_train.isnull().sum()
df_test.isnull().sum()
df_train.sample(10)
df_test.sample(10)
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)
df_train['HomePlanet'].unique()
df_train['HomePlanet'].fillna(df_train['HomePlanet'].mode(), inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].mode(), inplace=True)
pd.get_dummies(df_train['HomePlanet']).sum().plot.bar(rot=0, color='darkorange', figsize=(15, 7))
sns.countplot(data=df_train, x='HomePlanet', hue='Transported')
df_train = pd.concat([df_train, pd.get_dummies(df_train['HomePlanet'])], axis=1).drop(columns={'HomePlanet'})
df_test = pd.concat([df_test, pd.get_dummies(df_test['HomePlanet'])], axis=1).drop(columns={'HomePlanet'})
pd.get_dummies(df_train['CryoSleep']).sum().plot.bar(rot=0, color='skyblue', figsize=(15, 7))
sns.countplot(data=df_train, x='CryoSleep', hue='Transported')
df_train = pd.concat([df_train, pd.get_dummies(df_train['CryoSleep'])], axis=1).rename(columns={True: 'cryosleep', False: 'no_cryosleeper'}).drop(columns={'CryoSleep', 'no_cryosleeper'})
df_test = pd.concat([df_test, pd.get_dummies(df_test['CryoSleep'])], axis=1).rename(columns={True: 'cryosleep', False: 'no_cryosleeper'}).drop(columns={'CryoSleep', 'no_cryosleeper'})
df_train['Cabin'].fillna('x/0/x', inplace=True)
df_test['Cabin'].fillna('x/0/x', inplace=True)
df_train['Side'] = [df_train['Cabin'].str.split('/')[x][2] for x in range(len(df_train))]
df_test['Side'] = [df_test['Cabin'].str.split('/')[x][2] for x in range(len(df_test))]
pd.get_dummies(df_train['Side']).sum().plot.bar(rot=0, color='darkgrey', figsize=(15, 7))
sns.countplot(data=df_train, x='Side', hue='Transported')
df_train = pd.concat([df_train, pd.get_dummies(df_train['Side'])], axis=1).drop(columns={'Side', 'x', 'Cabin', 'P'}).rename(columns={'S': 'starboard_cabin'})
df_test = pd.concat([df_test, pd.get_dummies(df_test['Side'])], axis=1).drop(columns={'Side', 'x', 'Cabin', 'P'}).rename(columns={'S': 'starboard_cabin'})
df_train['Destination'].unique()
df_train['Destination'].fillna(df_train['Destination'].mode(), inplace=True)
df_test['Destination'].fillna(df_test['Destination'].mode(), inplace=True)
df_train = pd.concat([df_train, pd.get_dummies(df_train['Destination'])], axis=1).drop(columns={'Destination'}).rename(columns={'TRAPPIST-1e': 'destination_trappist', 'PSO J318.5-22': 'destination_pso', '55 Cancri e': 'destination_cancri'})
df_test = pd.concat([df_test, pd.get_dummies(df_test['Destination'])], axis=1).drop(columns={'Destination'}).rename(columns={'TRAPPIST-1e': 'destination_trappist', 'PSO J318.5-22': 'destination_pso', '55 Cancri e': 'destination_cancri'})
df_train['Age'].plot.hist(bins=20, color='lightgreen', figsize=(15, 7))
df_train['Age'].fillna(np.random.normal(df_train['Age'].mean(), df_train['Age'].std()), inplace=True)
df_test['Age'].fillna(np.random.normal(df_train['Age'].mean(), df_train['Age'].std()), inplace=True)
df_train['Age'].plot.hist(bins=20, color='darkgreen', figsize=(15, 7))
scaler = MinMaxScaler()
df_train['norm_age'] = scaler.fit_transform(np.array(df_train['Age']).reshape(-1, 1))
df_train.drop(columns={'Age'}, inplace=True)
df_test['norm_age'] = scaler.fit_transform(np.array(df_test['Age']).reshape(-1, 1))
df_test.drop(columns={'Age'}, inplace=True)
pd.get_dummies(df_train['VIP']).sum().plot.bar(color='silver', figsize=(15, 7))
df_train = pd.concat([df_train, pd.get_dummies(df_train['VIP'])], axis=1).drop(columns={'VIP', False}).rename(columns={True: 'VIP'})
df_test = pd.concat([df_test, pd.get_dummies(df_test['VIP'])], axis=1).drop(columns={'VIP', False}).rename(columns={True: 'VIP'})
df_train['luxury_expenses'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck']
df_test['luxury_expenses'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck']
df_train.drop(columns={'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'}, inplace=True)
df_test.drop(columns={'Name', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'}, inplace=True)
df_train['luxury_expenses'].mask((df_train['VIP'] == 1) & df_train['luxury_expenses'], df_train['luxury_expenses'].loc[df_train['VIP'] == 1].mean(), inplace=True)
df_train['luxury_expenses'].fillna(0.0, inplace=True)
df_test['luxury_expenses'].mask((df_train['VIP'] == 1) & df_train['luxury_expenses'], df_train['luxury_expenses'].loc[df_train['VIP'] == 1].mean(), inplace=True)
df_test['luxury_expenses'].fillna(0.0, inplace=True)
df_train['norm_luxury_expenses'] = scaler.fit_transform(np.array(df_train['luxury_expenses']).reshape(-1, 1))
df_test['norm_luxury_expenses'] = scaler.fit_transform(np.array(df_test['luxury_expenses']).reshape(-1, 1))
df_train.drop(columns={'luxury_expenses'}, inplace=True)
df_test.drop(columns={'luxury_expenses'}, inplace=True)
df_train = pd.concat([df_train, pd.get_dummies(df_train['Transported'])], axis=1).drop(columns={'Transported', False}).rename(columns={True: 'transported'})
df_train.head()
df_test.head()
df_X = df_train.drop('transported', axis=1)
df_y = df_train['transported']
random_forest = RandomForestClassifier(max_depth=10, random_state=132)
(X_train, X_test, y_train, y_test) = train_test_split(df_X, df_y, test_size=0.3, random_state=0)