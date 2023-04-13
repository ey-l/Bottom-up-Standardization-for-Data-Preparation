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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input0.head()
plt.figure(figsize=(8, 4))
sns.barplot(x='HomePlanet', y='Transported', data=_input1)
plt.figure(figsize=(8, 4))
sns.barplot(x='Destination', y='Transported', data=_input1)
_input1['Destination'].unique()
_input1 = _input1.replace({'Europa': 7, 'Earth': 4, 'Mars': 5}, inplace=False)
_input1 = _input1.replace({'TRAPPIST-1e': 4, 'PSO J318.5-22': 5, '55 Cancri e': 6}, inplace=False)
_input1 = _input1.replace({False: 0, True: 1}, inplace=False)
_input0 = _input0.replace({'Europa': 7, 'Earth': 4, 'Mars': 5}, inplace=False)
_input0 = _input0.replace({'TRAPPIST-1e': 4, 'PSO J318.5-22': 5, '55 Cancri e': 6}, inplace=False)
_input0 = _input0.replace({False: 0, True: 1}, inplace=False)
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1, inplace=False)
print(_input1.shape)
print(_input0.shape)
_input1.Transported.value_counts()
_input1.isnull().sum()
_input1 = _input1.fillna(_input1.median())
_input0 = _input0.fillna(_input0.median())
_input1 = _input1.fillna(method='bfill')
_input0 = _input0.fillna(method='bfill')
_input1.groupby('Transported').mean().T
_input1.info()
_input1.nunique()
_input1.corrwith(_input1['Transported']).abs().sort_values(ascending=False)
_input1.corr()
sns.heatmap(_input1.corr().abs(), cmap='Blues_r')
_input1['Cab0'] = _input1['Cabin'].str[0]
_input0['Cab0'] = _input0['Cabin'].str[0]
_input1['Cab-1'] = _input1['Cabin'].str[-1]
_input0['Cab-1'] = _input0['Cabin'].str[-1]
_input1.head()
_input1['Cab0'].unique()
X = _input1.copy()
y = _input1.Transported
X = X.drop('Transported', axis=1)
X.info()
X.columns
cat_attr = ['Cabin', 'Cab0', 'Cab-1']
num_attr = ['HomePlanet', 'CryoSleep', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Destination', 'VIP']
full_pip = ColumnTransformer([('num', StandardScaler(), num_attr), ('cat', OrdinalEncoder(), cat_attr)])
x = full_pip.fit_transform(X)
x_test = full_pip.fit_transform(_input0)
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2, random_state=41)
javob = []
for N in range(5):
    model = RandomForestClassifier(random_state=N, n_jobs=6, n_estimators=126)