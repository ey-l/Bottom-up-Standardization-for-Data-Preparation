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
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
sample_submission.head()
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head(5)
train.isnull().sum()
train['HomePlanet'].value_counts()
print(train['Cabin'].str[:1].unique())
print(train['Cabin'].str[2:3].unique())
print(train['Cabin'].str[-1:].unique())
train['Cabin'].str[-1:].value_counts()
train['Cabin'].str.split('/', expand=True)
train['Destination'].value_counts()
print('min : {}, max : {}, median : {}'.format(train['Age'].min(), train['Age'].max(), train['Age'].median()))
print('age distribution')
age_dist = pd.cut(train['Age'], list(range(0, 100, 10))).value_counts()
age_dist
name_df = train['Name'].str.split(' ', expand=True)
name_df = name_df.rename(columns={0: 'FirstName', 1: 'FamilyName'})
name_df
name_df['FamilyName'].value_counts()
train.describe()
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')
train_x = train.drop(columns=['Transported'])
train_y = train['Transported']
test_x = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_index = test_x['PassengerId']
train_x = train_x.drop(columns=['PassengerId', 'Name'])
test_x = test_x.drop(columns=['PassengerId', 'Name'])
train_x['Cabin'] = train_x['Cabin'].str[-1:]
test_x['Cabin'] = test_x['Cabin'].str[-1:]
train_x['CryoSleep'] = train_x['CryoSleep'].replace({True: 1, False: -1})
train_x['CryoSleep'] = train_x['CryoSleep'].fillna(0)
test_x['CryoSleep'] = test_x['CryoSleep'].replace({True: 1, False: -1})
test_x['CryoSleep'] = test_x['CryoSleep'].fillna(0)
train_x = pd.get_dummies(train_x, columns=['HomePlanet', 'Cabin', 'Destination'])
test_x = pd.get_dummies(test_x, columns=['HomePlanet', 'Cabin', 'Destination'])
train_x['VIP'] *= 1
train_x['VIP'] = train_x['VIP'].fillna(0)
test_x['VIP'] *= 1
test_x['VIP'] = test_x['VIP'].fillna(0)
train_x.head()
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=20, random_state=71)