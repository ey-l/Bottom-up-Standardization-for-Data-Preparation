import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
combine = [_input1, _input0]
combine
_input1.info()
_input1.describe()
_input1.nunique()
(_input1['HomePlanet'].value_counts(), _input1['Destination'].value_counts())
_input1['Cabin_Side'] = _input1.Cabin.str[-1]
_input1
_input1.drop(['Cabin'], axis=1)
_input1.describe(include=['O'])
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['Cabin_Side', 'Transported']].groupby(['Cabin_Side'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df = _input1
df
columns = 'RoomService\tFoodCourt\tShoppingMall\tSpa\tVRDeck\t'.split()
df['Expenses'] = df[columns].sum(axis=1)
df = df.drop(columns, axis=1)
df['Expenses'].describe()
df['Expenses_Band'] = pd.cut(df.Expenses, bins=[-1, 10000, 20000, 30000, 40000], labels=['(0-10k)', '(10k-20k)', '(20k-30k)', '(30k-40k)'])
df
df['Expenses_Band'].value_counts()
df[['Expenses_Band', 'Transported']].groupby(['Expenses_Band'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df['AgeBand'] = pd.cut(df['Age'], [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['(0-10)', '(10-20)', '(20-30)', '(30-40)', '(40-50)', '(50-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)'])
df = df.drop(['Name'], axis=1)
df
label = df['Transported']
df = df.drop(['Transported'], axis=1)
df
df = df.drop(['Age', 'Expenses', 'Cabin'], axis=1)
df = df[['PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expenses_Band', 'AgeBand', 'Cabin_Side']]
_input1 = df
df
col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expenses_Band', 'AgeBand', 'Cabin_Side']
for i in col:
    df = pd.get_dummies(df, columns=[i], drop_first=True)
columns = 'RoomService\tFoodCourt\tShoppingMall\tSpa\tVRDeck'.split()
_input0['Expenses'] = _input0[columns].sum(axis=1)
_input0 = _input0.drop(columns, axis=1)
_input0['Expenses_Band'] = pd.cut(_input0.Expenses, bins=[-1, 10000, 20000, 30000, 40000], labels=['(0-10k)', '(10k-20k)', '(20k-30k)', '(30k-40k)'])
_input0['AgeBand'] = pd.cut(_input0['Age'], [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['(0-10)', '(10-20)', '(20-30)', '(30-40)', '(40-50)', '(50-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)'])
_input0 = _input0.drop(['Name', 'Age'], axis=1)
_input0 = _input0.drop(['Expenses'], axis=1)
_input0
_input0['Cabin_Side'] = _input0.Cabin.str[-1]
_input0 = _input0.drop(['Cabin'], axis=1)
_input1.columns
_input0.columns
col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expenses_Band', 'AgeBand', 'Cabin_Side']
for i in col:
    _input0 = pd.get_dummies(_input0, columns=[i], drop_first=True)
_input0
df = df.drop(['PassengerId'], axis=1)
_input0.columns
df.columns
X_train = df
Y_train = label
X_test = _input0.drop('PassengerId', axis=1).copy()
(X_train.shape, Y_train.shape, X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression()