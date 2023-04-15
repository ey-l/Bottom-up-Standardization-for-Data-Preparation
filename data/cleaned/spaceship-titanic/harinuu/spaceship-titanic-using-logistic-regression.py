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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
train.info()
train.skew()
train.describe()
train.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
q1 = train['VIP'].quantile(0.25)
q2 = train['VIP'].quantile(0.75)
q1
train['VIP'] = np.where(train['VIP'] < q1, q1, train['VIP'])
train['VIP'] = np.where(train['VIP'] > q2, q2, train['VIP'])
train.skew()
train.head()
train.isnull().sum()
train.replace('-', 'nan')
train.replace('na', 'nan')
train['HomePlanet'].fillna(train['HomePlanet'].mode()[0], inplace=True)
train['CryoSleep'].fillna(train['CryoSleep'].mode()[0], inplace=True)
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
train = train.drop(['Cabin'], axis=1)
train.head()
train['Deck'].fillna(train['Deck'].mode()[0], inplace=True)
train['Num'].fillna(train['Num'].mode()[0], inplace=True)
train['Destination'].fillna(train['Destination'].mode()[0], inplace=True)
train.isnull().sum()
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Side'].fillna(train['Side'].mode()[0], inplace=True)
train.isnull().sum()
train.drop(['VIP'], axis=1, inplace=True)
train.head()
train['CryoSleep'].replace({True: 1, False: 0}, inplace=True)
train['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2}, inplace=True)
train['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=True)
train['Side'].replace({'P': 1, 'S': 0}, inplace=True)
train['Deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=True)
train.head()
cor = train.corr()
cor
font = {'size': 24, 'family': 'normal'}
mlt.rc('font', **font)
plt.figure(figsize=(20, 12))
sns.heatmap(cor, annot=True, cmap='coolwarm')

train.drop(['CryoSleep', 'Num'], axis=1, inplace=True)
train.head()
x_train = train.drop('Transported', axis=1)
y_train = train['Transported']
x_train
y_train
test.head()
test.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'VIP'], axis=1, inplace=True)
test.head()
test.isnull().sum()
test['HomePlanet'].fillna(test['HomePlanet'].mode()[0], inplace=True)
test['CryoSleep'].fillna(test['CryoSleep'].mode()[0], inplace=True)
test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', expand=True)
test.drop(['Cabin'], axis=1, inplace=True)
test.head()
test['Deck'].fillna(test['Deck'].mode()[0], inplace=True)
test.drop(['Num'], axis=1, inplace=True)
test['Side'].fillna(test['Side'].mode()[0], inplace=True)
test['Destination'].fillna(test['Destination'].mode()[0], inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test.skew()
test.drop(['CryoSleep'], axis=1, inplace=True)
test['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2}, inplace=True)
test['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}, inplace=True)
test['Deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=True)
test['Side'].replace({'P': 1, 'S': 0}, inplace=True)
test.head()
x_test = test.iloc[:, 0:]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()