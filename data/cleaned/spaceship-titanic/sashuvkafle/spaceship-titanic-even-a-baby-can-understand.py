import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_test.head()
df_train.describe()
df_train.dtypes
for i in df_train.columns:
    print(i, df_train[i].isna().sum())
df_train.dtypes
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = df_train[categorical_columns].mode().iloc[0]
df_train[categorical_columns] = df_train[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = df_train[numerical_columns].median()
df_train[numerical_columns] = df_train[numerical_columns].fillna(median)
for i in df_train.columns:
    print(i, df_train[i].isna().sum())
df_train = df_train.drop(columns=['Name'])
import seaborn as sns
import matplotlib.pyplot as plt
corr = df_train.corr()
(fig, ax) = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, ax=ax)

df_train['Expenses'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck']
df_train = df_train.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
df_train.head()
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
df_train['Age Group'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
df_train.head()
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['Age Group'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['HomePlanet'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['Transported'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['CryoSleep'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
string = df_train['Cabin'].str.split('/')
df_train['Deck'] = string.map(lambda string: string[0])
df_train['Number'] = string.map(lambda string: string[1])
df_train['Side'] = string.map(lambda string: string[2])
string = df_train['PassengerId'].str.split('_')
df_train['Group'] = string.map(lambda string: string[0])
df_train['Psngr_Num'] = string.map(lambda string: string[1])
df_train = df_train.drop(columns=['Cabin', 'PassengerId'])
df_train
df_train['Deck'].unique()
df_train['Side'].unique()
df_train['Number'].unique()
df_train['Psngr_Num'].unique()
df_train['Group'].unique()
df_train = df_train.drop(columns=['Number', 'Group'])
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['Deck'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
(fig, ax) = plt.subplots(figsize=(12, 7))
group_counts = df_train['Side'].value_counts()
sns.barplot(x=group_counts.index, y=group_counts.values)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_train['Transported'] = encoder.fit_transform(df_train['Transported'])
df_train['CryoSleep'] = encoder.fit_transform(df_train['CryoSleep'])
df_train['HomePlanet'] = encoder.fit_transform(df_train['HomePlanet'])
df_train['Age Group'] = encoder.fit_transform(df_train['Age Group'])
df_train['Destination'] = encoder.fit_transform(df_train['Destination'])
df_train['VIP'] = encoder.fit_transform(df_train['VIP'])
df_train['Side'] = encoder.fit_transform(df_train['Side'])
df_train['Deck'] = encoder.fit_transform(df_train['Deck'])
df_train.head()
corr = df_train.corr()
(fig, ax) = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, ax=ax)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_train[['Age', 'Expenses']] = ss.fit_transform(df_train[['Age', 'Expenses']])
df_train.head()
X_Train = df_train.drop('Transported', axis=1)
Y_Train = df_train['Transported']
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
(X_train, X_test, y_train, y_test) = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=0)
logreg = LogisticRegression()