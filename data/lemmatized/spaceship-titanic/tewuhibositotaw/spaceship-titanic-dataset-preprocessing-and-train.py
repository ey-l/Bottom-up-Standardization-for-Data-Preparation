import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input0.info()
_input1.describe()
_input0.describe()
msno.matrix(_input1)
msno.matrix(_input0)
_input1[['CryoSleep', 'VIP', 'Transported']] = (_input1[['CryoSleep', 'VIP', 'Transported']] == True).astype(int)
_input0[['CryoSleep', 'VIP']] = (_input0[['CryoSleep', 'VIP']] == True).astype(int)
data_df = pd.concat([_input1, _input0], axis=0)
data_df
msno.matrix(data_df)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Europa', inplace=False)
data_df['CryoSleep'] = data_df['CryoSleep'].fillna('False', inplace=False)
data_df['Cabin'] = data_df['Cabin'].fillna('X0000', inplace=False)
data_df['Destination'] = data_df['Destination'].fillna('55 Cancri e', inplace=False)
data_df['VIP'] = data_df['VIP'].fillna('False', inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['FoodCourt'] = data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=False)
data_df['ShoppingMall'] = data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=False)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
data_df['Name'] = data_df['Name'].fillna('Name not known', inplace=False)
msno.matrix(data_df)
train = data_df[0:len(_input1)]
test = data_df[len(_input1):]
train.info()
data_oh = pd.get_dummies(data_df, columns=['HomePlanet', 'Destination'])
data_df
data_oh
data_oh = data_oh.drop(['PassengerId', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=False)
data_oh
from sklearn.preprocessing import StandardScaler
num_cols = ['Age']
data_oh.describe()
data_std = data_oh.copy()
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
train = data_std[0:len(_input1)]
test = data_std[len(_input1):]
y = train['Transported']
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.info()
y_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()