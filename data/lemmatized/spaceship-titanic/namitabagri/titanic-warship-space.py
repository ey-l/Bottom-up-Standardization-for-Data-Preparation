import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1.shape)
_input1.describe()
_input1.head(3)
nun = _input1.isnull().sum()
nun
_input1.columns
_input1 = _input1.dropna(axis=0, inplace=False)
print('New shape of dataset after droping NaN values:  \n', _input1.shape)
print(type(_input1))
_input1.describe()
y = _input1['Transported']
y = y.dropna(axis=0)
print(y.shape)
features = ['PassengerId', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
x = _input1[features]
x = x.dropna(axis=0)
print(x.shape)
_input1.head(3)
print(x.head(2), ' \n \n', y.head(2))
x.head(2)
x.loc[:, ['Age']]
y.head()
import matplotlib.pyplot as plt
plt.title('x vs y')
plt.xlabel('criteria')
plt.ylabel('Transportation')
plt.scatter(x.loc[:, ['Age']], y)
from sklearn.model_selection import train_test_split
(train_x, val_x, train_y, val_y) = train_test_split(x, y, random_state=0)
print('train x', train_x.shape, '\n', 'train y', train_y.shape, '\n val x', val_x.shape, '\n valy', val_y.shape)
train_x = train_x.dropna(axis=0)
train_x.shape
train_y.head(3)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.shape
_input0.head(3)
test_features = ['PassengerId', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
test_x = _input0[test_features]
print(test_x.shape)
test_x.describe()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)