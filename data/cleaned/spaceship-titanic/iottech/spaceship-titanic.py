import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.columns
print(df.shape)
df.describe(include='all')
df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
df.head()
df.isnull().sum()
df.shape
df = df.fillna(0)
df.isnull().sum()
print(df.shape)

from statistics import stdev
sd = stdev(df['Age'])
sd2 = stdev(df['RoomService'])
sd3 = stdev(df['FoodCourt'])
sd4 = stdev(df['ShoppingMall'])
sd5 = stdev(df['Spa'])
sd6 = stdev(df['VRDeck'])
x_mean = df['Age'].mean()
x_mean1 = df['RoomService'].mean()
x_mean2 = df['FoodCourt'].mean()
x_mean3 = df['ShoppingMall'].mean()
x_mean4 = df['Spa'].mean()
x_mean5 = df['VRDeck'].mean()
df.head()
df.info()
df.info()
df['HomePlanet'].replace(['Europa', 'Earth', 'Mars', 0], [1, 2, 3, 0], inplace=True)
df['Destination'].replace(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 0], [1, 2, 3, 0], inplace=True)
df['VIP'].replace([False, True], [0, 1], inplace=True)
df['CryoSleep'].replace([False, True], [0, 1], inplace=True)
df['Transported'].replace([False, True], [0, 1], inplace=True)
sns.histplot(x='Age', data=df)
sns.countplot(x='HomePlanet', hue='Transported', data=df)
g = sns.FacetGrid(df, col='VIP', hue='Transported')
g.map_dataframe(sns.scatterplot, x='Spa', y='RoomService')
g = sns.FacetGrid(df, col='Transported', hue='VIP')
g.map_dataframe(sns.scatterplot, x='Age', y='FoodCourt')
g.add_legend()
g = sns.FacetGrid(df, col='HomePlanet', hue='Transported')
g.map_dataframe(sns.scatterplot, x='Spa', y='ShoppingMall')
g.add_legend()
sns.countplot(x='HomePlanet', data=df, hue='Transported')
X = df.drop(['Transported'], axis=1)
Y = df['Transported']
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