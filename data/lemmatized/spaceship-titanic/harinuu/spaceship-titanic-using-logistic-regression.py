import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input1.info()
_input1.skew()
_input1.describe()
_input1 = _input1.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=False)
q1 = _input1['VIP'].quantile(0.25)
q2 = _input1['VIP'].quantile(0.75)
q1
_input1['VIP'] = np.where(_input1['VIP'] < q1, q1, _input1['VIP'])
_input1['VIP'] = np.where(_input1['VIP'] > q2, q2, _input1['VIP'])
_input1.skew()
_input1.head()
_input1.isnull().sum()
_input1.replace('-', 'nan')
_input1.replace('na', 'nan')
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0], inplace=False)
_input1[['Deck', 'Num', 'Side']] = _input1['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop(['Cabin'], axis=1)
_input1.head()
_input1['Deck'] = _input1['Deck'].fillna(_input1['Deck'].mode()[0], inplace=False)
_input1['Num'] = _input1['Num'].fillna(_input1['Num'].mode()[0], inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['Side'] = _input1['Side'].fillna(_input1['Side'].mode()[0], inplace=False)
_input1.isnull().sum()
_input1 = _input1.drop(['VIP'], axis=1, inplace=False)
_input1.head()
_input1['CryoSleep'] = _input1['CryoSleep'].replace({True: 1, False: 0}, inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2}, inplace=False)
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=False)
_input1['Side'] = _input1['Side'].replace({'P': 1, 'S': 0}, inplace=False)
_input1['Deck'] = _input1['Deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=False)
_input1.head()
cor = _input1.corr()
cor
font = {'size': 24, 'family': 'normal'}
mlt.rc('font', **font)
plt.figure(figsize=(20, 12))
sns.heatmap(cor, annot=True, cmap='coolwarm')
_input1 = _input1.drop(['CryoSleep', 'Num'], axis=1, inplace=False)
_input1.head()
x_train = _input1.drop('Transported', axis=1)
y_train = _input1['Transported']
x_train
y_train
_input0.head()
_input0 = _input0.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'VIP'], axis=1, inplace=False)
_input0.head()
_input0.isnull().sum()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0], inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0], inplace=False)
_input0[['Deck', 'Num', 'Side']] = _input0['Cabin'].str.split('/', expand=True)
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input0.head()
_input0['Deck'] = _input0['Deck'].fillna(_input0['Deck'].mode()[0], inplace=False)
_input0 = _input0.drop(['Num'], axis=1, inplace=False)
_input0['Side'] = _input0['Side'].fillna(_input0['Side'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0], inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0.skew()
_input0 = _input0.drop(['CryoSleep'], axis=1, inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2}, inplace=False)
_input0['Destination'] = _input0['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=False)
_input0['Deck'] = _input0['Deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=False)
_input0['Side'] = _input0['Side'].replace({'P': 1, 'S': 0}, inplace=False)
_input0.head()
x_test = _input0.iloc[:, 0:]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()