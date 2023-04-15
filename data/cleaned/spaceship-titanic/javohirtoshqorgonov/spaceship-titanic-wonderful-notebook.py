import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.svm import SVC
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
test.head()
plt.figure(figsize=(8, 4))
sns.barplot(x='HomePlanet', y='Transported', data=train)

plt.figure(figsize=(8, 4))
sns.barplot(x='Destination', y='Transported', data=train)

train['Destination'].unique()
train.replace({'Europa': 7, 'Earth': 4, 'Mars': 5}, inplace=True)
train.replace({'TRAPPIST-1e': 4, 'PSO J318.5-22': 5, '55 Cancri e': 6}, inplace=True)
train.replace({False: 0, True: 1}, inplace=True)
test.replace({'Europa': 7, 'Earth': 4, 'Mars': 5}, inplace=True)
test.replace({'TRAPPIST-1e': 4, 'PSO J318.5-22': 5, '55 Cancri e': 6}, inplace=True)
test.replace({False: 0, True: 1}, inplace=True)
train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name'], axis=1, inplace=True)
print(train.shape)
print(test.shape)
train.Transported.value_counts()
train.isnull().sum()
train = train.fillna(train.median())
test = test.fillna(test.median())
train = train.fillna(method='bfill')
test = test.fillna(method='bfill')
train.groupby('Transported').mean().T
train.info()
train.nunique()
train.corrwith(train['Transported']).abs().sort_values(ascending=False)
train.corr()
sns.heatmap(train.corr().abs(), cmap='Blues_r')
train['Cab0'] = train['Cabin'].str[0]
test['Cab0'] = test['Cabin'].str[0]
train['Cab-1'] = train['Cabin'].str[-1]
test['Cab-1'] = test['Cabin'].str[-1]
train.head()
train['Cab0'].unique()
X = train.copy()
y = train.Transported
X = X.drop('Transported', axis=1)
X.info()
X.columns
cat_attr = ['Cabin', 'Cab0', 'Cab-1']
num_attr = ['HomePlanet', 'CryoSleep', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Destination', 'VIP']
full_pip = ColumnTransformer([('num', StandardScaler(), num_attr), ('cat', OrdinalEncoder(), cat_attr)])
x = full_pip.fit_transform(X)
x_test = full_pip.fit_transform(test)
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2, random_state=41)
javob = []
for N in range(5):
    model = RandomForestClassifier(random_state=N, n_jobs=6, n_estimators=126)