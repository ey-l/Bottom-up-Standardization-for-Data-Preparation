import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import time
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')


missingno.matrix(train_df, color=(0.06, 0.3, 0.57), fontsize=20)
missingno.matrix(test_df, color=(0.06, 0.3, 0.57), fontsize=20)
train_df_2 = train_df.drop(['Age', 'FoodCourt', 'ShoppingMall'], axis=1)
test_df_2 = test_df.drop(['Age', 'FoodCourt', 'ShoppingMall'], axis=1)


data_df = pd.concat([train_df, test_df], axis=0)
data_df_2 = pd.concat([train_df_2, test_df_2], axis=0)
data_df
data_df_2
data_df['HomePlanet'].fillna('Unknown', inplace=True)
data_df_2['HomePlanet'].fillna('Unknown', inplace=True)
data_df['CryoSleep'].fillna('Unknown', inplace=True)
data_df_2['CryoSleep'].fillna('Unknown', inplace=True)
data_df['Destination'].fillna('Unknown', inplace=True)
data_df_2['Destination'].fillna('Unknown', inplace=True)
data_df['Cabin'].fillna('X', inplace=True)
data_df_2['Cabin'].fillna('X', inplace=True)
data_df['Cabin'].value_counts()
data_df_2['Cabin'].value_counts()
data_df['Deck'] = data_df['Cabin'].str[0]
data_df_2['Deck'] = data_df_2['Cabin'].str[0]
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights='distance')
data_df['Age'] = imputer.fit_transform(data_df[['Age']])
data_df['VIP'] = imputer.fit_transform(data_df[['VIP']])
data_df_2['VIP'] = imputer.fit_transform(data_df_2[['VIP']])
data_df['RoomService'] = imputer.fit_transform(data_df[['RoomService']])
data_df_2['RoomService'] = imputer.fit_transform(data_df_2[['RoomService']])
data_df['RoomService'] = imputer.fit_transform(data_df[['RoomService']])
data_df_2['RoomService'] = imputer.fit_transform(data_df_2[['RoomService']])
data_df['FoodCourt'] = imputer.fit_transform(data_df[['FoodCourt']])
data_df['ShoppingMall'] = imputer.fit_transform(data_df[['ShoppingMall']])
data_df['Spa'] = imputer.fit_transform(data_df[['Spa']])
data_df_2['Spa'] = imputer.fit_transform(data_df_2[['Spa']])
data_df['VRDeck'] = imputer.fit_transform(data_df[['VRDeck']])
data_df_2['VRDeck'] = imputer.fit_transform(data_df_2[['VRDeck']])
data_oh = pd.get_dummies(data_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck'])
data_oh_2 = pd.get_dummies(data_df_2, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck'])
data_oh.drop(['Name', 'Cabin'], axis=1, inplace=True)
data_oh_2.drop(['Name', 'Cabin'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
num_cols_2 = ['RoomService', 'Spa', 'VRDeck']
data_oh.describe()
data_oh_2.describe()
data_std = data_oh.copy()
data_std_2 = data_oh_2.copy()
scaler = StandardScaler()
data_std[num_cols] = scaler.fit_transform(data_oh[num_cols])
data_std_2[num_cols_2] = scaler.fit_transform(data_oh_2[num_cols_2])
data_std.describe()
data_std_2.describe()
train = data_std[0:len(train_df)]
test = data_std[len(train_df):]
train_2 = data_std_2[0:len(train_df)]
test_2 = data_std_2[len(train_df):]
y = train['Transported'].astype('int')
X = train.drop('Transported', axis=1)
y_2 = train_2['Transported'].astype('int')
X_2 = train_2.drop('Transported', axis=1)
X_test = test.drop('Transported', axis=1)
X_test_2 = test_2.drop('Transported', axis=1)
X_test
X_test_2
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=32)
(X_train_2, X_val_2, y_train_2, y_val_2) = train_test_split(X_2, y_2, test_size=0.2, random_state=32)
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
lr = LogisticRegression()
dt = tree.DecisionTreeClassifier()
gnb = GaussianNB()
neigh = KNeighborsClassifier()
clf = svm.SVC()
lr2 = LogisticRegression()
dt2 = tree.DecisionTreeClassifier()
gnb2 = GaussianNB()
neigh2 = KNeighborsClassifier()
clf2 = svm.SVC()
start_time = time.time()