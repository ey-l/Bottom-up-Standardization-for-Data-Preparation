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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.describe()
_input1.describe(include='object')
_input0.head()
_input0.shape
_input0.info()
_input0.describe()
_input0.describe(include='object')
total = _input1.isnull().sum()
percent = total / _input1.isnull().count() * 100
pd.DataFrame({'total': total.sort_values(ascending=False), 'percent': percent.sort_values(ascending=False)})
total = _input0.isnull().sum()
percent = total / _input0.isnull().count() * 100
pd.DataFrame({'total': total.sort_values(ascending=False), 'percent': percent.sort_values(ascending=False)})
total = float(_input1.shape[0])
ax = sns.countplot(x='Transported', data=_input1)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.0, height + 5, '{:1.2f}'.format(height / total * 100), ha='center')
category_cols = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['Age'].hist()
_input1['RoomService'].hist()
_input1['FoodCourt'].hist()
_input1['ShoppingMall'].hist()
_input1['Spa'].hist()
_input1['VRDeck'].hist()
total = len(_input1)
ax = sns.countplot(x='HomePlanet', data=_input1)
for p in ax.patches:
    height = p.get_height()
    width = p.get_width() / 2
    ax.text(p.get_x() + width, height + 5, '{:.2f}'.format(height / total * 100), ha='center')
ax = sns.countplot(x='CryoSleep', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='Destination', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='VIP', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
sns.boxplot(x='Transported', y='Age', data=_input1)
sns.boxplot(x='Transported', y='RoomService', data=_input1)
sns.boxplot(x='Transported', y='FoodCourt', data=_input1)
sns.boxplot(x='Transported', y='ShoppingMall', data=_input1)
sns.boxplot(x='Transported', y='Spa', data=_input1)
sns.boxplot(x='Transported', y='VRDeck', data=_input1)
ax = sns.countplot(x='HomePlanet', hue='Transported', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='CryoSleep', hue='Transported', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='Destination', hue='Transported', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='VIP', hue='Transported', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
corr = _input1[numerical_cols].corr()
sns.heatmap(data=corr, annot=True)
sns.heatmap(data=_input1.corr(), annot=True)
cabin_train = _input1['Cabin'].astype('category')
_input1['cabin_group'] = cabin_train.apply(lambda x: x.split('/')[0])
cabin_test = _input0['Cabin'].astype('category')
_input0['cabin_group'] = cabin_test.apply(lambda x: x.split('/')[0])
_input1 = _input1.drop(columns='Cabin', axis=1)
_input0 = _input0.drop(columns='Cabin', axis=1)
_input1.head()
_input1['cabin_group'].unique()
ax = sns.countplot(x='cabin_group', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
ax = sns.countplot(x='cabin_group', hue='Transported', data=_input1)
for p in ax.patches:
    (w, h) = (p.get_width(), p.get_height())
    x = p.get_x() + w / 2
    y = h + 5
    percent = h / len(_input1) * 100
    ax.text(x, y, '{:.2f}'.format(percent), ha='center')
billed_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['total_billed'] = _input1[billed_cols].sum(axis=1)
_input0['total_billed'] = _input0[billed_cols].sum(axis=1)
_input1 = _input1.drop(columns=billed_cols, axis=1)
_input0 = _input0.drop(columns=billed_cols, axis=1)
_input1.head()
_input1['total_billed'].hist()
_input1['total_billed'] = _input1['total_billed'].apply(lambda x: np.log(1 + x))
_input0['total_billed'] = _input0['total_billed'].apply(lambda x: np.log(1 + x))
_input1['total_billed'].hist()
sns.boxplot(y='total_billed', x='Transported', data=_input1)
home_planet_map = {'Europa': 1, 'Earth': 2, 'Mars': 3}
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0], inplace=False)
_input1['HomePlanet'] = _input1['HomePlanet'].map(home_planet_map)
_input0['HomePlanet'] = _input0['HomePlanet'].map(home_planet_map)
cryoSleep_map = {False: 0, True: 1}
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0], inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].map(cryoSleep_map)
_input0['CryoSleep'] = _input0['CryoSleep'].map(cryoSleep_map)
destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0], inplace=False)
_input1['Destination'] = _input1['Destination'].map(destination_map)
_input0['Destination'] = _input0['Destination'].map(destination_map)
vip_map = {False: 0, True: 1}
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0], inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0], inplace=False)
_input1['VIP'] = _input1['VIP'].map(vip_map)
_input0['VIP'] = _input0['VIP'].map(vip_map)
cabin_group_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': '8'}
_input1['cabin_group'] = _input1['cabin_group'].fillna(_input1['cabin_group'].mode()[0], inplace=False)
_input0['cabin_group'] = _input0['cabin_group'].fillna(_input0['cabin_group'].mode()[0], inplace=False)
_input1['cabin_group'] = _input1['cabin_group'].map(cabin_group_map)
_input0['cabin_group'] = _input0['cabin_group'].map(cabin_group_map)
transported_map = {False: 0, True: 1}
_input1['Transported'] = _input1['Transported'].map(transported_map)
_input1.head()
_input1.dtypes
_input1['cabin_group'] = _input1['cabin_group'].astype('int64')
_input0['cabin_group'] = _input0['cabin_group'].astype('int64')
_input1.dtypes
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input1['total_billed'] = _input1['total_billed'].fillna(_input1['total_billed'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
_input0['total_billed'] = _input0['total_billed'].fillna(_input0['total_billed'].median(), inplace=False)
_input1.isnull().sum()
_input1 = _input1.drop(columns=['Name', 'PassengerId', 'VIP'], axis=1)
test_ids = _input0['PassengerId']
_input0 = _input0.drop(columns=['Name', 'PassengerId', 'VIP'], axis=1)
y = _input1['Transported']
X = _input1.drop(columns='Transported')
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.3, random_state=42)
print('train shape:', X_train.shape)
print('valid shape:', X_valid.shape)
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
df_model = pd.DataFrame(columns=['model', 'valid_score', 'train_score', 'precision', 'recall', 'f1'])

def model_scores(name, model):
    global df_model