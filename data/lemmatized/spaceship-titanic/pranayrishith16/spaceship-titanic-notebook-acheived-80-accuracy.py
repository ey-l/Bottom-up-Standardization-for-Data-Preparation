import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('white')
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
data1 = _input1.copy(deep=False)
data1
data1 = data1.set_index('PassengerId', inplace=False)
data1.info()
data1.describe()
data1.describe(include=('object', 'bool'))
data1.isna().sum()
cols = data1.columns
for col in cols:
    if data1[col].dtype != 'float64':
        print(data1[col].value_counts())
data1['RoomService'] = data1['RoomService'].fillna(data1['RoomService'].median())
data1['FoodCourt'] = data1['FoodCourt'].fillna(data1['FoodCourt'].median())
data1['ShoppingMall'] = data1['ShoppingMall'].fillna(data1['ShoppingMall'].median())
data1['VRDeck'] = data1['VRDeck'].fillna(data1['VRDeck'].median())
data1['Spa'] = data1['Spa'].fillna(data1['Spa'].median())
data1['CryoSleep'] = data1['CryoSleep'].astype(bool)
data1['VIP'] = data1['VIP'].astype(bool)
data1['HomePlanet'] = data1['HomePlanet'].fillna('Earth')
data1['Destination'] = data1['Destination'].fillna('TRAPPIST-1e')
data1['CryoSleep'] = data1['CryoSleep'].fillna(False)
data1['VIP'] = data1['VIP'].fillna(False)
data1['Age'] = data1['Age'].fillna(data1['Age'].mean())
data1['Cabin'] = data1['Cabin'].fillna('B/0/S')
bins = [0, 14, 24, 64, 99]
labels = ['children', 'youth', 'adult', 'senior']
data1['Age_group'] = pd.cut(data1['Age'], bins=bins, labels=labels, right=False)
data1 = data1.drop('Age', inplace=False, axis='columns')
data1['Age_group']
data1['Total_spending'] = data1['RoomService'] + data1['FoodCourt'] + data1['ShoppingMall'] + data1['VRDeck'] + data1['Spa']
data1 = data1.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa'], axis='columns')
data1['Cabin_Side'] = data1['Cabin'].str.split('/').str[2]
data1['Cabin_Deck'] = data1['Cabin'].str.split('/').str[0]
data1 = data1.drop('Cabin', axis=1)
data1 = data1.dropna()
data1 = data1.drop('Name', axis='columns', inplace=False)
data1.columns
sns.barplot(x='Transported', y='HomePlanet', data=data1)
sns.barplot(x='Transported', y='Destination', data=data1)
sns.barplot(x='Transported', y='CryoSleep', data=data1)
sns.barplot(x='Transported', y='VIP', data=data1)
sns.barplot(x='Transported', y='Cabin_Deck', data=data1)
sns.barplot(x='Transported', y='Cabin_Side', data=data1)
data1['CryoSleep'] = data1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
data1['Transported'] = data1['Transported'].replace({False: 0, True: 1}, inplace=False)
data1['VIP'] = data1['VIP'].replace({False: 0, True: 1}, inplace=False)
f = plt.figure(figsize=(24, 20))
sns.heatmap(data1.corr(), annot=True, vmin=0, vmax=1)
data1['HomePlanet'] = data1['HomePlanet'].astype('category')
data1['Destination'] = data1['Destination'].astype('category')
data1['Cabin_Deck'] = data1['Cabin_Deck'].astype('category')
data1['Cabin_Side'] = data1['Cabin_Side'].astype('category')
data1['Cabin_Side'] = labelencoder.fit_transform(data1['Cabin_Side'])
data1['Cabin_Deck'] = labelencoder.fit_transform(data1['Cabin_Deck'])
data1 = pd.get_dummies(data1, columns=['HomePlanet', 'Destination', 'Age_group'])
data1.columns
f = plt.figure(figsize=(24, 20))
sns.heatmap(data1.corr(), annot=True, vmin=0, vmax=1)
Y = data1['Transported']
X = data1.drop('Transported', axis='columns')
data1.info()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import catboost as ctb
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3, random_state=41)
rfc = RandomForestClassifier(n_estimators=1000)