import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(df.shape)
df.describe()
df.head(3)
nun = df.isnull().sum()
nun
df.columns
df = df.dropna(axis=0, inplace=False)
print('New shape of dataset after droping NaN values:  \n', df.shape)
print(type(df))
df.describe()
y = df['Transported']
y = y.dropna(axis=0)
print(y.shape)
features = ['PassengerId', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
x = df[features]
x = x.dropna(axis=0)
print(x.shape)
df.head(3)
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
testdf = pd.read_csv('data/input/spaceship-titanic/test.csv')
testdf.shape
testdf.head(3)
test_features = ['PassengerId', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
test_x = testdf[test_features]
print(test_x.shape)
test_x.describe()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)