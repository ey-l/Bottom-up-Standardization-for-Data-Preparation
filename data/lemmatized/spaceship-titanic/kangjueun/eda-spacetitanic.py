import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import missingno
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1
_input0
_input1.info()
_input0.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
plt.figure(figsize=(10, 5))
sns.heatmap(_input1.isna().T, cmap='Purples')
plt.title('missing values')
pd.isnull(_input1).sum()
_input1.info()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
_input1[num_cols].corr()
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='YlGnBu')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
sns.kdeplot(data=_input1, x='Age', hue='Transported')
_input1['Age'] = _input1['Age'].fillna(-1)
bins = [-2, 7, 20, 40, 70, np.inf]
labels = ['unknown', 'children', 'young people', 'adult', 'senior']
_input1['AgeGroup'] = pd.cut(_input1['Age'], bins, labels=labels)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
sns.countplot(data=_input1, x='AgeGroup', hue='Transported')
_input1['luxury amenities'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1['luxury amenities'].max()
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
sns.kdeplot(data=_input1, x='luxury amenities', hue='Transported')
ax.set_xlim(0, 30000)
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
sns.barplot(x='HomePlanet', y='Transported', data=_input1)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
('Crosleep transported success ratio', _input1['Transported'][_input1['CryoSleep'] == True].value_counts(normalize=True)[1])
('Crosleep transported failed ratio', _input1['Transported'][_input1['CryoSleep'] == False].value_counts(normalize=True)[1])
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
sns.barplot(x='HomePlanet', y='Transported', data=_input1)
_input1.groupby(by=['HomePlanet', 'Destination']).count()
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
sns.countplot(data=_input1, x='VIP', hue='Transported')
('VIP transported success ratio', _input1['Transported'][_input1['VIP'] == True].value_counts(normalize=True)[1])
('VIP transported failed ratio', _input1['Transported'][_input1['VIP'] == False].value_counts(normalize=True)[1])
_input1['Name'].nunique()
_input1['Name'].unique()
_input1['Cabin'].nunique()
_input1['Cabin'].unique()
_input1['Cabin'].value_counts()
_input1['deck'] = _input1['Cabin'].str[0]
_input1['deck'].value_counts()
_input1.groupby(by=['deck']).count()
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
sns.barplot(data=_input1, x='deck', y='Transported', palette='Oranges')
trains = _input1.copy()
data = pd.concat([trains, _input0], axis=0)
cols = data.columns[data.isna().any()].tolist()
missingvalue = pd.DataFrame(data[cols].isna().sum(), columns=['Number_missing'])
missingvalue['Percentage_missing'] = np.round(100 * missingvalue['Number_missing'] / len(data), 2)
missingvalue
print('Europa')
Eu = data[data['HomePlanet'] == 'Europa'].shape[0]
print(Eu)
print('Earth')
Ea = data[data['HomePlanet'] == 'Earth'].shape[0]
print(Ea)
print('Mars')
M = data[data['HomePlanet'] == 'Mars'].shape[0]
print(M)
data = data.fillna({'HomePlanet': 'Earth'})
HomePlanet_mapping = {'Europa': 1, 'Earth': 2, 'Mars': 3}
data['HomePlanet'] = data['HomePlanet'].map(HomePlanet_mapping)
_input0['HomePlanet'] = _input0['HomePlanet'].map(HomePlanet_mapping)
data
print('TRAPPIST-1e')
T = data[data['Destination'] == 'TRAPPIST-1e'].shape[0]
print(T)
print('PSO J318.5-22')
P = data[data['Destination'] == 'PSO J318.5-22'].shape[0]
print(P)
print('55 Cancri e')
C = data[data['Destination'] == '55 Cancri e'].shape[0]
print(C)
data = data.fillna({'Destination': 'TRAPPIST-1e'})
Destination_mapping = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
data['Destination'] = data['Destination'].map(Destination_mapping)
_input0['Destination'] = _input0['Destination'].map(Destination_mapping)
data
print('True')
T = data[data['CryoSleep'] == True].shape[0]
print(T)
print('False')
F = data[data['CryoSleep'] == False].shape[0]
print(F)
data = data.fillna({'CryoSleep': False})
CryoSleep_mapping = {True: 0, False: 1}
data['CryoSleep'] = data['CryoSleep'].map(CryoSleep_mapping)
_input0['CryoSleep'] = _input0['CryoSleep'].map(CryoSleep_mapping)
data
print('True')
T = data[data['VIP'] == True].shape[0]
print(T)
print('False')
F = data[data['VIP'] == False].shape[0]
print(F)
data = data.fillna({'VIP': False})
VIP_mapping = {True: 0, False: 1}
data['VIP'] = data['VIP'].map(VIP_mapping)
_input0['VIP'] = _input0['VIP'].map(VIP_mapping)
data
Transported_mapping = {True: 0, False: 1}
data['Transported'] = data['Transported'].map(Transported_mapping)
data['Age'] = data['Age'].fillna(data['Age'].mean(), inplace=False)
data['luxury amenities'] = data['luxury amenities'].fillna(data['luxury amenities'].mean(), inplace=False)
data.info()
data = data.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'deck', 'AgeGroup'], axis=1, inplace=False)
data
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
data_oh = pd.get_dummies(data, columns=cat_cols)
data_oh
data_oh.describe()
num_cols = ['Age', 'luxury amenities']
from sklearn.preprocessing import StandardScaler
data_std = data_oh.copy()
data_oh.describe()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
data_oh.describe()
data_std.describe()
from sklearn.preprocessing import MinMaxScaler
data_minmax = data_oh.copy()
scaler = MinMaxScaler()
data_minmax[num_cols] = scaler.fit_transform(data_minmax[num_cols])
data_oh.describe()
data_minmax.describe()
data_std
missingno.matrix(data_std)
train = data_std[0:len(trains)]
test = data_std[len(trains):]
train.info()
test.info()
train
y = train['Transported']
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
X
X_test
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.5, random_state=47)
X_train
X_val
y_train
y_val
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()