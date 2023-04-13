import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1.describe()
_input1.head()
_input1 = _input1.drop(['Name', 'PassengerId'], axis=1, inplace=False)
_input1.isnull().sum()
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='HomePlanet', hue='Destination', data=_input1)
plt.xlabel('DIFFERENT HOME PLANETS', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['CryoSleep'].value_counts()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0], inplace=False)
_input1['Age'].value_counts()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mode()[0], inplace=False)
_input1['VIP'].value_counts()
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0], inplace=False)
cabin_data = _input1['Cabin'].str.split('/', n=2, expand=True)
cabin_data
_input1['Deck'] = cabin_data[0]
_input1['Number'] = cabin_data[1]
_input1['Side'] = cabin_data[2]
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input1.head()
_input1['Deck'] = _input1['Deck'].fillna(_input1['Deck'].mode()[0], inplace=False)
_input1['Side'] = _input1['Side'].fillna(_input1['Side'].mode()[0], inplace=False)
_input1['Number'] = _input1['Number'].fillna(_input1['Number'].mode()[0], inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(), inplace=False)
_input1.isnull().sum()
_input1.head()
lst = [_input1]
for column in lst:
    column.loc[(column['Age'] >= 0) & (column['Age'] <= 1), 'Age_category'] = 'Infant'
    column.loc[(column['Age'] >= 2) & (column['Age'] <= 4), 'Age_category'] = 'Toddler'
    column.loc[(column['Age'] >= 5) & (column['Age'] <= 12), 'Age_category'] = 'Child'
    column.loc[(column['Age'] >= 13) & (column['Age'] <= 19), 'Age_category'] = 'Teen'
    column.loc[(column['Age'] >= 20) & (column['Age'] <= 39), 'Age_category'] = 'Adult'
    column.loc[(column['Age'] >= 40) & (column['Age'] <= 59), 'Age_category'] = 'Middle Age Adult'
    column.loc[column['Age'] >= 60, 'Age_category'] = 'Senior Adult'
_input1['Age_category'] = _input1['Age_category'].astype('category')
_input1 = _input1.drop('Age', axis=1, inplace=False)
new_data = _input1.copy()
lst_of_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported', 'Deck', 'Side', 'Age_category']
for i in lst_of_col:
    new_data[i] = new_data[i].astype('category')
    new_data[i] = new_data[i].cat.codes
label_encoded_data = new_data.copy()
new_data['Number'].value_counts()
lst_of_col_two = ['HomePlanet', 'Destination', 'Deck', 'Age_category']
for i in lst_of_col_two:
    new_data = pd.get_dummies(new_data, columns=[i], drop_first=True)
new_data.head()
X = new_data.drop('Transported', axis=1)
y = new_data['Transported']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled
X_scaled = pd.DataFrame(X_scaled)
X_scaled
y.value_counts()
_input1.head()
plt.figure(figsize=(15, 8))
corr = _input1.corr()
sns.heatmap(corr, annot=True)
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='Deck', hue='HomePlanet', data=_input1, palette='spring')
plt.xticks(fontsize=15)
plt.xlabel('DIFFERENT DECKS', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
filter1 = _input1['Deck'].isin(['T'])
df1 = _input1[filter1]
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='CryoSleep', hue='Transported', data=_input1, palette='nipy_spectral')
plt.xticks(fontsize=15)
plt.xlabel('People transported or not', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
plt.figure(figsize=(10, 6))
plots = sns.countplot(x='Destination', hue='Transported', data=_input1, palette='mako')
plt.xticks(fontsize=10)
plt.xlabel('Different Destination Planets', fontsize=15)
plt.ylabel('Count', fontsize=15)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.25, random_state=111)
from xgboost import XGBClassifier
xgb = XGBClassifier()