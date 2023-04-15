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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.info()
test_df.info()


train_df.describe()
test_df.describe()
msno.matrix(train_df)
msno.matrix(test_df)
train_df[['CryoSleep', 'VIP', 'Transported']] = (train_df[['CryoSleep', 'VIP', 'Transported']] == True).astype(int)
test_df[['CryoSleep', 'VIP']] = (test_df[['CryoSleep', 'VIP']] == True).astype(int)
data_df = pd.concat([train_df, test_df], axis=0)
data_df
msno.matrix(data_df)
data_df['Age'].fillna(data_df['Age'].mean(), inplace=True)
data_df['HomePlanet'].fillna('Europa', inplace=True)
data_df['CryoSleep'].fillna('False', inplace=True)
data_df['Cabin'].fillna('X0000', inplace=True)
data_df['Destination'].fillna('55 Cancri e', inplace=True)
data_df['VIP'].fillna('False', inplace=True)
data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=True)
data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=True)
data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=True)
data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=True)
data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=True)
data_df['Name'].fillna('Name not known', inplace=True)
msno.matrix(data_df)
train = data_df[0:len(train_df)]
test = data_df[len(train_df):]
train.info()
data_oh = pd.get_dummies(data_df, columns=['HomePlanet', 'Destination'])
data_df
data_oh
data_oh.drop(['PassengerId', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
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
train = data_std[0:len(train_df)]
test = data_std[len(train_df):]
y = train['Transported']
X = train.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.info()
y_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()