import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1
_input0
_input1.info()
_input0.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
data_df = pd.concat([_input1, _input0], axis=0)
data_df
missingno.matrix(data_df)
data_df['Cabin'] = data_df['Cabin'].fillna('X/000/Y', inplace=False)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['FoodCourt'] = data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=False)
data_df['ShoppingMall'] = data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=False)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
data_df['Name'] = data_df['Name'].fillna('None', inplace=False)
data_df['VIP'] = data_df['VIP'].fillna('False', inplace=False)
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Earth', inplace=False)
data_df['Destination'] = data_df['Destination'].fillna('TRAPPIST-1e', inplace=False)
data_df['CryoSleep'] = data_df['CryoSleep'].fillna('False', inplace=False)
missingno.matrix(data_df)
data_df['PayedBill'] = data_df['RoomService'] + data_df['FoodCourt'] + data_df['ShoppingMall'] + data_df['Spa'] + data_df['VRDeck']
data_df
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PayedBill']
data_df[num_cols].corr()
data_df['Destination'].value_counts()
data_df['CryoSleep'].value_counts()
A = data_df[data_df['Destination'] == '55 Cancri e']
A['CryoSleep'].value_counts()
B = data_df[data_df['Destination'] == 'PSO J318.5-22']
B['CryoSleep'].value_counts()
C = data_df[data_df['Destination'] == 'TRAPPIST-1e']
C['CryoSleep'].value_counts()
D = data_df[['VIP', 'PayedBill']]
sns.catplot(data=D, x='VIP', y='PayedBill', kind='bar')
cond1 = data_df['PayedBill'] >= 2000
data_df[cond1]
sns.catplot(data=data_df, x='HomePlanet', y='PayedBill', kind='bar')
sns.catplot(data=data_df, x='Destination', y='PayedBill', kind='bar')
data_df['HomePlanet'].value_counts()
data_df['Name'].nunique()
sns.histplot(data=data_df, x='Age', kde=True, bins=20)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
sns.swarmplot(data=data_df, x='Destination', y='Age', size=5, ax=ax)
data_df = data_df.drop(['Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=False)
data_df
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
data_oh = pd.get_dummies(data_df, columns=cat_cols)
data_oh
data_oh.describe()
num_cols = ['Age', 'PayedBill']
from sklearn.preprocessing import StandardScaler
data_std = data_oh.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_std[num_cols])
from sklearn.preprocessing import MinMaxScaler
data_minmax = data_oh.copy()
scaler = MinMaxScaler()
data_minmax[num_cols] = scaler.fit_transform(data_minmax[num_cols])
data_oh.describe()
data_minmax.describe()
data_std
missingno.matrix(data_std)
train = data_std[0:len(_input1)]
test = data_std[len(_input1):]
train.info()
test.info()
train
y = train['Transported']
y = y.astype('int')
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=56)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.cluster import Birch
from sklearn.svm import LinearSVC
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn import metrics
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
model = LogisticRegression()