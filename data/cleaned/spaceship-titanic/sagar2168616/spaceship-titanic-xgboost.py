import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data.head()
data.info()
data.describe()
data.head()
data.drop(['Name', 'PassengerId'], axis=1, inplace=True)
data.isnull().sum()
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='HomePlanet', hue='Destination', data=data)
plt.xlabel('DIFFERENT HOME PLANETS', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
data['HomePlanet'].fillna('Earth', inplace=True)
data['Destination'].fillna('TRAPPIST-1e', inplace=True)
data['CryoSleep'].value_counts()
data['CryoSleep'].fillna(data['CryoSleep'].mode()[0], inplace=True)
data['Age'].value_counts()
data['Age'].fillna(data['Age'].mode()[0], inplace=True)
data['VIP'].value_counts()
data['VIP'].fillna(data['VIP'].mode()[0], inplace=True)
cabin_data = data['Cabin'].str.split('/', n=2, expand=True)
cabin_data
data['Deck'] = cabin_data[0]
data['Number'] = cabin_data[1]
data['Side'] = cabin_data[2]
data.drop('Cabin', axis=1, inplace=True)
data.head()
data['Deck'].fillna(data['Deck'].mode()[0], inplace=True)
data['Side'].fillna(data['Side'].mode()[0], inplace=True)
data['Number'].fillna(data['Number'].mode()[0], inplace=True)
data['RoomService'].fillna(data['RoomService'].median(), inplace=True)
data['FoodCourt'].fillna(data['FoodCourt'].median(), inplace=True)
data['ShoppingMall'].fillna(data['ShoppingMall'].median(), inplace=True)
data['Spa'].fillna(data['Spa'].median(), inplace=True)
data['VRDeck'].fillna(data['VRDeck'].median(), inplace=True)
data.isnull().sum()
data.head()
lst = [data]
for column in lst:
    column.loc[(column['Age'] >= 0) & (column['Age'] <= 1), 'Age_category'] = 'Infant'
    column.loc[(column['Age'] >= 2) & (column['Age'] <= 4), 'Age_category'] = 'Toddler'
    column.loc[(column['Age'] >= 5) & (column['Age'] <= 12), 'Age_category'] = 'Child'
    column.loc[(column['Age'] >= 13) & (column['Age'] <= 19), 'Age_category'] = 'Teen'
    column.loc[(column['Age'] >= 20) & (column['Age'] <= 39), 'Age_category'] = 'Adult'
    column.loc[(column['Age'] >= 40) & (column['Age'] <= 59), 'Age_category'] = 'Middle Age Adult'
    column.loc[column['Age'] >= 60, 'Age_category'] = 'Senior Adult'
data['Age_category'] = data['Age_category'].astype('category')
data.drop('Age', axis=1, inplace=True)
new_data = data.copy()
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
data.head()
plt.figure(figsize=(15, 8))
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='Deck', hue='HomePlanet', data=data, palette='spring')
plt.xticks(fontsize=15)
plt.xlabel('DIFFERENT DECKS', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
filter1 = data['Deck'].isin(['T'])
df1 = data[filter1]
plt.figure(figsize=(15, 8))
plots = sns.countplot(x='CryoSleep', hue='Transported', data=data, palette='nipy_spectral')
plt.xticks(fontsize=15)
plt.xlabel('People transported or not', fontsize=20)
plt.ylabel('COUNT', fontsize=20)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
plt.figure(figsize=(10, 6))
plots = sns.countplot(x='Destination', hue='Transported', data=data, palette='mako')
plt.xticks(fontsize=10)
plt.xlabel('Different Destination Planets', fontsize=15)
plt.ylabel('Count', fontsize=15)
for bar in plots.patches:
    plots.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.25, random_state=111)
from xgboost import XGBClassifier
xgb = XGBClassifier()