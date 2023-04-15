import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
space_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
space_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
space_train.shape
space_test.shape
space_train.head()
space_test.head()
space_train.info()
space_train.describe()
space_train.PassengerId.nunique() / space_train.shape[0]
space_train.Transported.value_counts()
space_train.isnull().sum()
space_test.isnull().sum()
space_train['HomePlanet'].fillna(method='ffill', inplace=True)
space_train['CryoSleep'].fillna(method='ffill', inplace=True)
space_train['Cabin'].fillna(method='ffill', inplace=True)
space_train['Destination'].fillna(method='ffill', inplace=True)
space_train['Age'].fillna(space_train['Age'].mean(), inplace=True)
space_train['VIP'].fillna(method='ffill', inplace=True)
space_train['RoomService'].fillna(space_train['RoomService'].mean(), inplace=True)
space_train['FoodCourt'].fillna(space_train['FoodCourt'].mean(), inplace=True)
space_train['ShoppingMall'].fillna(space_train['ShoppingMall'].mean(), inplace=True)
space_train['Spa'].fillna(space_train['Spa'].mean(), inplace=True)
space_train['VRDeck'].fillna(space_train['VRDeck'].mean(), inplace=True)
space_train['Name'].fillna(method='ffill', inplace=True)
space_train.isnull().sum()
space_test['HomePlanet'].fillna(method='ffill', inplace=True)
space_test['CryoSleep'].fillna(method='ffill', inplace=True)
space_test['Cabin'].fillna(method='ffill', inplace=True)
space_test['Destination'].fillna(method='ffill', inplace=True)
space_test['Age'].fillna(space_test['Age'].mean(), inplace=True)
space_test['VIP'].fillna(method='ffill', inplace=True)
space_test['RoomService'].fillna(space_test['RoomService'].mean(), inplace=True)
space_test['FoodCourt'].fillna(space_test['FoodCourt'].mean(), inplace=True)
space_test['ShoppingMall'].fillna(space_test['ShoppingMall'].mean(), inplace=True)
space_test['Spa'].fillna(space_test['Spa'].mean(), inplace=True)
space_test['VRDeck'].fillna(space_test['VRDeck'].mean(), inplace=True)
space_test['Name'].fillna(method='ffill', inplace=True)
space_test.isnull().sum()
plt.figure(figsize=(7, 7))
plt.title('Distribution of Transported Passengers')
space_train['Transported'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='HomePlanet', data=space_train)
plt.title('Transported successfully from Home Planet')
plt.figure(figsize=(7, 7))
plt.title(' Passengers Confined to Cabins')
space_train['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='CryoSleep', data=space_train)
plt.title('Confined Passengers Transported')
plt.figure(figsize=(7, 7))
space_train['Destination'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='Destination', data=space_train)
space_train['Side'] = space_train['Cabin'].str.split('/').str[2]
plt.figure(figsize=(7, 7))
space_train['Side'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Side', y='Transported', kind='bar', palette='mako', data=space_train)
space_train['Deck'] = space_train['Cabin'].str.split('/').str[0]
plt.figure(figsize=(7, 7))
space_train['Deck'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Transported', y='Deck', kind='bar', palette='ch:.25', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', palette='pastel', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Age', hue='Transported', kind='box', palette='viridis', data=space_train)
plt.figure(figsize=(7, 7))
space_train['VIP'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(10, 10))
sns.catplot(x='Destination', y='VIP', hue='Transported', kind='point', palette='Spectral', data=space_train)
space_train['Expenses'] = space_train['RoomService'] + space_train['FoodCourt'] + space_train['ShoppingMall'] + space_train['Spa'] + space_train['VRDeck']
plt.figure(figsize=(10, 10))
sns.catplot(x='VIP', y='Expenses', hue='Transported', kind='bar', palette='icefire', data=space_train)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Expenses', hue='Transported', kind='bar', palette='coolwarm', data=space_train)
sns.scatterplot(x='Age', y='Expenses', data=space_train[space_train.Transported == True])
plt.figure(figsize=(7, 7))
sns.barplot(x='Side', y='Expenses', palette='ch:s=-.2,r=.6', data=space_train)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for x in [i for i in space_train.columns if len(space_train[i].unique()) == 2]:
    print(x, space_train[x].unique())
    space_train[x] = label_encoder.fit_transform(space_train[x])
space_train.drop(['HomePlanet', 'Cabin', 'Destination', 'Name', 'Side', 'Expenses', 'Deck'], axis=1, inplace=True)
space_train.head()
[[x, space_train[x].unique()] for x in [i for i in space_train.columns if len(space_train[i].unique()) < 10]]
space_test.drop(['HomePlanet', 'Cabin', 'Destination', 'Name'], axis=1, inplace=True)
space_test.head()
label_encoder = LabelEncoder()
for x in [i for i in space_test.columns if len(space_test[i].unique()) == 2]:
    print(x, space_test[x].unique())
    space_test[x] = label_encoder.fit_transform(space_test[x])
[[x, space_test[x].unique()] for x in [i for i in space_test.columns if len(space_test[i].unique()) < 10]]
space_train.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
x_train = space_train.drop(['Transported', 'PassengerId'], axis=1)
x_test = space_test.drop(['PassengerId'], axis=1)
Y_train = space_train['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(x_train, Y_train, test_size=0.25, random_state=0)
print('shape of X_train:', X_train.shape)
print('shape of y_train:', y_train.shape[0])
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = pd.DataFrame(X_train, columns=x_train.columns)
X_test = pd.DataFrame(X_test, columns=x_test.columns)
model = LogisticRegression()