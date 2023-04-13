import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import json
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from xgboost import plot_tree
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2.head()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(5)
_input1.isnull().sum()
_input1['HomePlanet'].value_counts()
print(_input1['Cabin'].str[:1].unique())
print(_input1['Cabin'].str[2:3].unique())
print(_input1['Cabin'].str[-1:].unique())
_input1['Cabin'].str[-1:].value_counts()
_input1['Cabin'].str.split('/', expand=True)
_input1['Destination'].value_counts()
print('min : {}, max : {}, median : {}'.format(_input1['Age'].min(), _input1['Age'].max(), _input1['Age'].median()))
print('age distribution')
age_dist = pd.cut(_input1['Age'], list(range(0, 100, 10))).value_counts()
age_dist
name_df = _input1['Name'].str.split(' ', expand=True)
name_df = name_df.rename(columns={0: 'FirstName', 1: 'FamilyName'})
name_df
name_df['FamilyName'].value_counts()
_input1.describe()
corr = _input1.corr()
corr.style.background_gradient(cmap='coolwarm')
train_x = _input1.drop(columns=['Transported'])
train_y = _input1['Transported']
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_index = _input0['PassengerId']
train_x = train_x.drop(columns=['PassengerId', 'Name'])
_input0 = _input0.drop(columns=['PassengerId', 'Name'])
train_x['Cabin'] = train_x['Cabin'].str[-1:]
_input0['Cabin'] = _input0['Cabin'].str[-1:]
train_x['CryoSleep'] = train_x['CryoSleep'].replace({True: 1, False: -1})
train_x['CryoSleep'] = train_x['CryoSleep'].fillna(0)
_input0['CryoSleep'] = _input0['CryoSleep'].replace({True: 1, False: -1})
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(0)
train_x = pd.get_dummies(train_x, columns=['HomePlanet', 'Cabin', 'Destination'])
_input0 = pd.get_dummies(_input0, columns=['HomePlanet', 'Cabin', 'Destination'])
train_x['VIP'] *= 1
train_x['VIP'] = train_x['VIP'].fillna(0)
_input0['VIP'] *= 1
_input0['VIP'] = _input0['VIP'].fillna(0)
train_x.head()
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=20, random_state=71)