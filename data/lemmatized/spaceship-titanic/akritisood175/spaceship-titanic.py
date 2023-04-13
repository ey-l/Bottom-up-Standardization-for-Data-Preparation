import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.shape
_input0.shape
_input1.head()
_input0.head()
_input1.info()
_input1.describe()
_input1.PassengerId.nunique() / _input1.shape[0]
_input1.Transported.value_counts()
_input1.isnull().sum()
_input0.isnull().sum()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(method='ffill', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(method='ffill', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(method='ffill', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(method='ffill', inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(method='ffill', inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
_input1['Name'] = _input1['Name'].fillna(method='ffill', inplace=False)
_input1.isnull().sum()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(method='ffill', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(method='ffill', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna(method='ffill', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(method='ffill', inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(method='ffill', inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean(), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean(), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean(), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean(), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean(), inplace=False)
_input0['Name'] = _input0['Name'].fillna(method='ffill', inplace=False)
_input0.isnull().sum()
plt.figure(figsize=(7, 7))
plt.title('Distribution of Transported Passengers')
_input1['Transported'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='HomePlanet', data=_input1)
plt.title('Transported successfully from Home Planet')
plt.figure(figsize=(7, 7))
plt.title(' Passengers Confined to Cabins')
_input1['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='CryoSleep', data=_input1)
plt.title('Confined Passengers Transported')
plt.figure(figsize=(7, 7))
_input1['Destination'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.barplot(x='Transported', y='Destination', data=_input1)
_input1['Side'] = _input1['Cabin'].str.split('/').str[2]
plt.figure(figsize=(7, 7))
_input1['Side'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Side', y='Transported', kind='bar', palette='mako', data=_input1)
_input1['Deck'] = _input1['Cabin'].str.split('/').str[0]
plt.figure(figsize=(7, 7))
_input1['Deck'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(7, 7))
sns.catplot(x='Transported', y='Deck', kind='bar', palette='ch:.25', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', palette='pastel', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Age', hue='Transported', kind='box', palette='viridis', data=_input1)
plt.figure(figsize=(7, 7))
_input1['VIP'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.figure(figsize=(10, 10))
sns.catplot(x='Destination', y='VIP', hue='Transported', kind='point', palette='Spectral', data=_input1)
_input1['Expenses'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
plt.figure(figsize=(10, 10))
sns.catplot(x='VIP', y='Expenses', hue='Transported', kind='bar', palette='icefire', data=_input1)
plt.figure(figsize=(10, 10))
sns.catplot(x='HomePlanet', y='Expenses', hue='Transported', kind='bar', palette='coolwarm', data=_input1)
sns.scatterplot(x='Age', y='Expenses', data=_input1[_input1.Transported == True])
plt.figure(figsize=(7, 7))
sns.barplot(x='Side', y='Expenses', palette='ch:s=-.2,r=.6', data=_input1)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for x in [i for i in _input1.columns if len(_input1[i].unique()) == 2]:
    print(x, _input1[x].unique())
    _input1[x] = label_encoder.fit_transform(_input1[x])
_input1 = _input1.drop(['HomePlanet', 'Cabin', 'Destination', 'Name', 'Side', 'Expenses', 'Deck'], axis=1, inplace=False)
_input1.head()
[[x, _input1[x].unique()] for x in [i for i in _input1.columns if len(_input1[i].unique()) < 10]]
_input0 = _input0.drop(['HomePlanet', 'Cabin', 'Destination', 'Name'], axis=1, inplace=False)
_input0.head()
label_encoder = LabelEncoder()
for x in [i for i in _input0.columns if len(_input0[i].unique()) == 2]:
    print(x, _input0[x].unique())
    _input0[x] = label_encoder.fit_transform(_input0[x])
[[x, _input0[x].unique()] for x in [i for i in _input0.columns if len(_input0[i].unique()) < 10]]
_input1.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
x_train = _input1.drop(['Transported', 'PassengerId'], axis=1)
x_test = _input0.drop(['PassengerId'], axis=1)
Y_train = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(x_train, Y_train, test_size=0.25, random_state=0)
print('shape of X_train:', X_train.shape)
print('shape of y_train:', y_train.shape[0])
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = pd.DataFrame(X_train, columns=x_train.columns)
X_test = pd.DataFrame(X_test, columns=x_test.columns)
model = LogisticRegression()