import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df
combine = [train_df, test_df]
combine
train_df.info()
train_df.describe()
train_df.nunique()
(train_df['HomePlanet'].value_counts(), train_df['Destination'].value_counts())
train_df['Cabin_Side'] = train_df.Cabin.str[-1]
train_df
train_df.drop(['Cabin'], axis=1)
train_df.describe(include=['O'])
train_df[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['Cabin_Side', 'Transported']].groupby(['Cabin_Side'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_df[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
df = train_df
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
train_df = df
df
col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expenses_Band', 'AgeBand', 'Cabin_Side']
for i in col:
    df = pd.get_dummies(df, columns=[i], drop_first=True)
columns = 'RoomService\tFoodCourt\tShoppingMall\tSpa\tVRDeck'.split()
test_df['Expenses'] = test_df[columns].sum(axis=1)
test_df = test_df.drop(columns, axis=1)
test_df['Expenses_Band'] = pd.cut(test_df.Expenses, bins=[-1, 10000, 20000, 30000, 40000], labels=['(0-10k)', '(10k-20k)', '(20k-30k)', '(30k-40k)'])
test_df['AgeBand'] = pd.cut(test_df['Age'], [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['(0-10)', '(10-20)', '(20-30)', '(30-40)', '(40-50)', '(50-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)'])
test_df = test_df.drop(['Name', 'Age'], axis=1)
test_df = test_df.drop(['Expenses'], axis=1)
test_df
test_df['Cabin_Side'] = test_df.Cabin.str[-1]
test_df = test_df.drop(['Cabin'], axis=1)
train_df.columns
test_df.columns
col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expenses_Band', 'AgeBand', 'Cabin_Side']
for i in col:
    test_df = pd.get_dummies(test_df, columns=[i], drop_first=True)
test_df
df = df.drop(['PassengerId'], axis=1)
test_df.columns
df.columns
X_train = df
Y_train = label
X_test = test_df.drop('PassengerId', axis=1).copy()
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