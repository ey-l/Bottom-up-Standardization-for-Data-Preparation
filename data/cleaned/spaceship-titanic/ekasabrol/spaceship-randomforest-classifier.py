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

train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train = train.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
train.head(3)
plt.figure(figsize=(15, 10))
sns.heatmap(train.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})

categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for i in range(len(categorical)):
    train[categorical[i]] = train[categorical[i]].fillna(train[categorical[i]].value_counts().index[0])
continuous = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in range(len(continuous)):
    train[continuous[i]] = train[continuous[i]].fillna(round(train[continuous[i]].median(), 0))
    train[continuous[i]] = train[continuous[i]].astype('int64')
plt.figure(figsize=(15, 10))
sns.heatmap(train.isna().transpose(), cbar_kws={'label': 'Missing Data'})

binary = ['CryoSleep', 'VIP', 'Transported']
for i in range(len(binary)):
    train[binary[i]] = train[binary[i]] * 1
column = ['HomePlanet', 'Destination']

def get_one_hot_vectors(dataframe, column):
    for i in range(len(column)):
        y = pd.get_dummies(dataframe[column[i]], prefix=column[i])
        dataframe = dataframe.merge(y, how='outer', left_index=True, right_index=True)
        dataframe = dataframe.drop(column[i], axis=1)
    return dataframe
train = get_one_hot_vectors(train, column)
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test = test.drop(['Name', 'Cabin'], axis=1)
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for i in range(len(categorical)):
    test[categorical[i]] = test[categorical[i]].fillna(test[categorical[i]].value_counts().index[0])
continuous = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in range(len(continuous)):
    test[continuous[i]] = test[continuous[i]].fillna(round(test[continuous[i]].median(), 0))
    test[continuous[i]] = test[continuous[i]].astype('int64')
binary = ['CryoSleep', 'VIP']
for i in range(len(binary)):
    test[binary[i]] = test[binary[i]] * 1
column = ['HomePlanet', 'Destination']
test = get_one_hot_vectors(test, column)
test.head()
from sklearn.ensemble import RandomForestClassifier
X_train = train.loc[:, train.columns != 'Transported']
y_train = train['Transported']
X_test = test.loc[:, test.columns != 'PassengerId']
forest = RandomForestClassifier(random_state=0, max_depth=10, bootstrap=False, criterion='gini')