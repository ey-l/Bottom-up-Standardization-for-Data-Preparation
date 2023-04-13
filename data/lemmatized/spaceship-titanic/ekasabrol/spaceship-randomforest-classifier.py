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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1 = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
_input1.head(3)
plt.figure(figsize=(15, 10))
sns.heatmap(_input1.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for i in range(len(categorical)):
    _input1[categorical[i]] = _input1[categorical[i]].fillna(_input1[categorical[i]].value_counts().index[0])
continuous = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in range(len(continuous)):
    _input1[continuous[i]] = _input1[continuous[i]].fillna(round(_input1[continuous[i]].median(), 0))
    _input1[continuous[i]] = _input1[continuous[i]].astype('int64')
plt.figure(figsize=(15, 10))
sns.heatmap(_input1.isna().transpose(), cbar_kws={'label': 'Missing Data'})
binary = ['CryoSleep', 'VIP', 'Transported']
for i in range(len(binary)):
    _input1[binary[i]] = _input1[binary[i]] * 1
column = ['HomePlanet', 'Destination']

def get_one_hot_vectors(dataframe, column):
    for i in range(len(column)):
        y = pd.get_dummies(dataframe[column[i]], prefix=column[i])
        dataframe = dataframe.merge(y, how='outer', left_index=True, right_index=True)
        dataframe = dataframe.drop(column[i], axis=1)
    return dataframe
_input1 = get_one_hot_vectors(_input1, column)
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0 = _input0.drop(['Name', 'Cabin'], axis=1)
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for i in range(len(categorical)):
    _input0[categorical[i]] = _input0[categorical[i]].fillna(_input0[categorical[i]].value_counts().index[0])
continuous = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in range(len(continuous)):
    _input0[continuous[i]] = _input0[continuous[i]].fillna(round(_input0[continuous[i]].median(), 0))
    _input0[continuous[i]] = _input0[continuous[i]].astype('int64')
binary = ['CryoSleep', 'VIP']
for i in range(len(binary)):
    _input0[binary[i]] = _input0[binary[i]] * 1
column = ['HomePlanet', 'Destination']
_input0 = get_one_hot_vectors(_input0, column)
_input0.head()
from sklearn.ensemble import RandomForestClassifier
X_train = _input1.loc[:, _input1.columns != 'Transported']
y_train = _input1['Transported']
X_test = _input0.loc[:, _input0.columns != 'PassengerId']
forest = RandomForestClassifier(random_state=0, max_depth=10, bootstrap=False, criterion='gini')