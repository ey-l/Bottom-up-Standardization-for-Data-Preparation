import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_train.shape
df_train.info()
df_train.describe()
df_train.describe(include='object')
df_test.head()
df_test.shape
df_test.info()
df_test.describe()
df_test.describe(include='object')
total = df_train.isnull().sum()
percent = total / df_train.isnull().count() * 100
pd.DataFrame({'total': total.sort_values(ascending=False), 'percent': percent.sort_values(ascending=False)})
total = df_test.isnull().sum()
percent = total / df_test.isnull().count() * 100
pd.DataFrame({'total': total.sort_values(ascending=False), 'percent': percent.sort_values(ascending=False)})
total = float(df_train.shape[0])
ax = sns.countplot(x='Transported', data=df_train)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.0, height + 5, '{:1.2f}'.format(height / total * 100), ha='center')

category_cols = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_train['Age'].hist()
df_train['RoomService'].hist()
df_train['FoodCourt'].hist()
df_train['ShoppingMall'].hist()
df_train['Spa'].hist()
df_train['VRDeck'].hist()
total = len(df_train)
ax = sns.countplot(x='HomePlanet', data=df_train)
for p in ax.patches:
    height = p.get_height()
    width = p.get_width() / 2
    ax.text(p.get_x() + width, height + 5, '{:.2f}'.format(height / total * 100), ha='center')
ax = sns.countplot(x='CryoSleep', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='Destination', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='VIP', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
sns.boxplot(x='Transported', y='Age', data=df_train)
sns.boxplot(x='Transported', y='RoomService', data=df_train)
sns.boxplot(x='Transported', y='FoodCourt', data=df_train)
sns.boxplot(x='Transported', y='ShoppingMall', data=df_train)
sns.boxplot(x='Transported', y='Spa', data=df_train)
sns.boxplot(x='Transported', y='VRDeck', data=df_train)
ax = sns.countplot(x='HomePlanet', hue='Transported', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='CryoSleep', hue='Transported', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='Destination', hue='Transported', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='VIP', hue='Transported', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
corr = df_train[numerical_cols].corr()
sns.heatmap(data=corr, annot=True)
sns.heatmap(data=df_train.corr(), annot=True)
cabin_train = df_train['Cabin'].astype('category')
df_train['cabin_group'] = cabin_train.apply(lambda x: x.split('/')[0])
cabin_test = df_test['Cabin'].astype('category')
df_test['cabin_group'] = cabin_test.apply(lambda x: x.split('/')[0])
df_train = df_train.drop(columns='Cabin', axis=1)
df_test = df_test.drop(columns='Cabin', axis=1)
df_train.head()
df_train['cabin_group'].unique()
ax = sns.countplot(x='cabin_group', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='cabin_group', hue='Transported', data=df_train)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(df_train) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
billed_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_train['total_billed'] = df_train[billed_cols].sum(axis=1)
df_test['total_billed'] = df_test[billed_cols].sum(axis=1)
df_train = df_train.drop(columns=billed_cols, axis=1)
df_test = df_test.drop(columns=billed_cols, axis=1)
df_train.head()
df_train['total_billed'].hist()
df_train['total_billed'] = df_train['total_billed'].apply(lambda x: np.log(1 + x))
df_test['total_billed'] = df_test['total_billed'].apply(lambda x: np.log(1 + x))
df_train['total_billed'].hist()
sns.boxplot(y='total_billed', x='Transported', data=df_train)
home_planet_map = {'Europa': 1, 'Earth': 2, 'Mars': 3}
df_train['HomePlanet'].fillna(df_train['HomePlanet'].mode()[0], inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].mode()[0], inplace=True)
df_train['HomePlanet'] = df_train['HomePlanet'].map(home_planet_map)
df_test['HomePlanet'] = df_test['HomePlanet'].map(home_planet_map)
cryoSleep_map = {False: 0, True: 1}
df_train['CryoSleep'].fillna(df_train['CryoSleep'].mode()[0], inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].mode()[0], inplace=True)
df_train['CryoSleep'] = df_train['CryoSleep'].map(cryoSleep_map)
df_test['CryoSleep'] = df_test['CryoSleep'].map(cryoSleep_map)
destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
df_train['Destination'].fillna(df_train['Destination'].mode()[0], inplace=True)
df_test['Destination'].fillna(df_test['Destination'].mode()[0], inplace=True)
df_train['Destination'] = df_train['Destination'].map(destination_map)
df_test['Destination'] = df_test['Destination'].map(destination_map)
vip_map = {False: 0, True: 1}
df_train['VIP'].fillna(df_train['VIP'].mode()[0], inplace=True)
df_test['VIP'].fillna(df_test['VIP'].mode()[0], inplace=True)
df_train['VIP'] = df_train['VIP'].map(vip_map)
df_test['VIP'] = df_test['VIP'].map(vip_map)
cabin_group_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': '8'}
df_train['cabin_group'].fillna(df_train['cabin_group'].mode()[0], inplace=True)
df_test['cabin_group'].fillna(df_test['cabin_group'].mode()[0], inplace=True)
df_train['cabin_group'] = df_train['cabin_group'].map(cabin_group_map)
df_test['cabin_group'] = df_test['cabin_group'].map(cabin_group_map)
transported_map = {False: 0, True: 1}
df_train['Transported'] = df_train['Transported'].map(transported_map)
df_train.head()
df_train.dtypes
df_train['cabin_group'] = df_train['cabin_group'].astype('int64')
df_test['cabin_group'] = df_test['cabin_group'].astype('int64')
df_train.dtypes
df_train.isnull().sum()
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['total_billed'].fillna(df_train['total_billed'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['total_billed'].fillna(df_test['total_billed'].median(), inplace=True)
df_train.isnull().sum()
df_train = df_train.drop(columns=['Name', 'PassengerId', 'VIP'], axis=1)
test_ids = df_test['PassengerId']
df_test = df_test.drop(columns=['Name', 'PassengerId', 'VIP'], axis=1)
y = df_train['Transported']
X = df_train.drop(columns='Transported')
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.3, random_state=42)
print('train shape:', X_train.shape)
print('valid shape:', X_valid.shape)
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
df_model = pd.DataFrame(columns=['model', 'valid_score', 'train_score', 'precision', 'recall', 'f1'])

def model_scores(name, model):
    global df_model