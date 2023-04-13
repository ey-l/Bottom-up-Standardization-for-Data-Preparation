import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.columns
print(_input1.shape)
_input1.describe(include='all')
_input1 = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=False)
_input1.head()
_input1.isnull().sum()
_input1.shape
_input1 = _input1.fillna(0)
_input1.isnull().sum()
print(_input1.shape)
from statistics import stdev
sd = stdev(_input1['Age'])
sd2 = stdev(_input1['RoomService'])
sd3 = stdev(_input1['FoodCourt'])
sd4 = stdev(_input1['ShoppingMall'])
sd5 = stdev(_input1['Spa'])
sd6 = stdev(_input1['VRDeck'])
x_mean = _input1['Age'].mean()
x_mean1 = _input1['RoomService'].mean()
x_mean2 = _input1['FoodCourt'].mean()
x_mean3 = _input1['ShoppingMall'].mean()
x_mean4 = _input1['Spa'].mean()
x_mean5 = _input1['VRDeck'].mean()
_input1.head()
_input1.info()
_input1.info()
_input1['HomePlanet'] = _input1['HomePlanet'].replace(['Europa', 'Earth', 'Mars', 0], [1, 2, 3, 0], inplace=False)
_input1['Destination'] = _input1['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 0], [1, 2, 3, 0], inplace=False)
_input1['VIP'] = _input1['VIP'].replace([False, True], [0, 1], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace([False, True], [0, 1], inplace=False)
_input1['Transported'] = _input1['Transported'].replace([False, True], [0, 1], inplace=False)
sns.histplot(x='Age', data=_input1)
sns.countplot(x='HomePlanet', hue='Transported', data=_input1)
g = sns.FacetGrid(_input1, col='VIP', hue='Transported')
g.map_dataframe(sns.scatterplot, x='Spa', y='RoomService')
g = sns.FacetGrid(_input1, col='Transported', hue='VIP')
g.map_dataframe(sns.scatterplot, x='Age', y='FoodCourt')
g.add_legend()
g = sns.FacetGrid(_input1, col='HomePlanet', hue='Transported')
g.map_dataframe(sns.scatterplot, x='Spa', y='ShoppingMall')
g.add_legend()
sns.countplot(x='HomePlanet', data=_input1, hue='Transported')
X = _input1.drop(['Transported'], axis=1)
Y = _input1['Transported']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.33)
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
Tree = tree.DecisionTreeClassifier()
Log_regg = LogisticRegression(random_state=2)
gaussian_svm = GaussianNB()
g_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)