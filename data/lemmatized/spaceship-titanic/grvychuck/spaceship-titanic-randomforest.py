import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
warnings.simplefilter(action='ignore')
_input1.head(80)
_input0.head()
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1.shape
_input0.shape
_input1.info()
_input0.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.describe()
_input1.describe(include=['O'])
_input1.hist(figsize=(15, 10))
train_test = [_input1, _input0]
for dataset in train_test:
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['RoomService'] = dataset['RoomService'].fillna(dataset['RoomService'].median())
    dataset['FoodCourt'] = dataset['FoodCourt'].fillna(dataset['FoodCourt'].median())
    dataset['Spa'] = dataset['Spa'].fillna(dataset['Spa'].median())
    dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(dataset['ShoppingMall'].median())
    dataset['VRDeck'] = dataset['VRDeck'].fillna(dataset['VRDeck'].median())
for dataset in train_test:
    dataset['HomePlanet'] = dataset['HomePlanet'].fillna(dataset['HomePlanet'].mode()[0])
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(dataset['CryoSleep'].mode()[0])
    dataset['Cabin'] = dataset['Cabin'].fillna(dataset['Cabin'].mode()[0])
    dataset['Destination'] = dataset['Destination'].fillna(dataset['Destination'].mode()[0])
    dataset['VIP'] = dataset['VIP'].fillna(dataset['VIP'].mode()[0])
_input1.isnull().sum()
_input0.isnull().sum()
for dataset in train_test:
    dataset = dataset.drop('Name', axis=1, inplace=False)
_input1.head()
for dataset in train_test:
    dataset['CryoSleep'] = dataset['CryoSleep'].astype(int)
    dataset['VIP'] = dataset['VIP'].astype(int)
_input1.head()
_input0.head()
temp_Cabin_train = _input1['Cabin'].str.split('/', expand=True).rename(columns={0: 'Cabin_0', 1: 'Cabin_1', 2: 'Cabin_2'})
_input1 = pd.concat([_input1, temp_Cabin_train], axis=1)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
temp_Cabin_test = _input0['Cabin'].str.split('/', expand=True).rename(columns={0: 'Cabin_0', 1: 'Cabin_1', 2: 'Cabin_2'})
_input0 = pd.concat([_input0, temp_Cabin_test], axis=1)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1.head()
_input0.head()
train_test = [_input1, _input0]
print('Age min', _input1['Age'].min())
print('Age max', _input1['Age'].max())
_input1['Age'].hist()
for dataset in train_test:
    dataset.loc[(dataset['Age'] >= 0) & (dataset['Age'] < 10), 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 10) & (dataset['Age'] < 20), 'Age'] = 10
    dataset.loc[(dataset['Age'] >= 20) & (dataset['Age'] < 30), 'Age'] = 20
    dataset.loc[(dataset['Age'] >= 30) & (dataset['Age'] < 40), 'Age'] = 30
    dataset.loc[(dataset['Age'] >= 40) & (dataset['Age'] < 50), 'Age'] = 40
    dataset.loc[(dataset['Age'] >= 50) & (dataset['Age'] < 60), 'Age'] = 50
    dataset.loc[(dataset['Age'] >= 60) & (dataset['Age'] < 70), 'Age'] = 60
    dataset.loc[dataset['Age'] >= 70, 'Age'] = 70
_input1.head()
plt.figure(figsize=[10, 10])
plt.subplot(2, 2, 1)
sns.countplot(data=_input1, x='HomePlanet', hue='Transported')
plt.title('HomePlanet vs. Transported')
plt.subplot(2, 2, 2)
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
plt.title('CryoSleep vs. Transported')
plt.subplot(2, 2, 3)
sns.countplot(data=_input1, x='Destination', hue='Transported')
plt.title('Destination vs. Transported')
plt.subplot(2, 2, 4)
sns.countplot(data=_input1, x='VIP', hue='Transported')
plt.title('VIP vs. Transported')
plt.subplots_adjust(left=0, right=1.5, wspace=0.3, hspace=0.3)

def bar_chart(feature):
    Transported = _input1[_input1['Transported'] == 1][feature].value_counts()
    Not = _input1[_input1['Transported'] == 0][feature].value_counts()
    df = pd.DataFrame([Transported, Not])
    df.index = ['Transported', 'Not']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))

def bar_non_stacked(feature):
    Transported = _input1[_input1['Transported'] == 1][feature].value_counts()
    Not = _input1[_input1['Transported'] == 0][feature].value_counts()
    df = pd.DataFrame([Transported, Not])
    df.index = ['Transported', 'Not']
    df.plot(kind='bar', stacked=False, figsize=(10, 5))
bar_chart('HomePlanet')
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('CryoSleep')
_input1[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Destination')
_input1[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Age')
bar_non_stacked('Age')
sns.countplot(data=_input1, x='Age', hue='Transported')
plt.title('Age vs. Transported')
_input1[['Age', 'Transported']].groupby(['Age'], as_index=False).mean()
bar_chart('VIP')
_input1[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1
bar_non_stacked('Cabin_0')
_input1[['Cabin_0', 'Transported']].groupby(['Cabin_0'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Cabin_2')
_input1[['Cabin_2', 'Transported']].groupby(['Cabin_2'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1['HomePlanet'].value_counts()
planet_mapping = {'Earth': 0, 'Europa': 1, 'Mars': 2}
for dataset in train_test:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(planet_mapping)
_input1['Destination'].value_counts()
dest_mapping = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
for dataset in train_test:
    dataset['Destination'] = dataset['Destination'].map(dest_mapping)
_input1['Cabin_0'].value_counts()
cabin0_mapping = {'A': 0, 'B': 0.3, 'C': 0.6, 'D': 0.9, 'E': 1.2, 'F': 1.5, 'G': 1.8, 'T': 2.1}
for dataset in train_test:
    dataset['Cabin_0'] = dataset['Cabin_0'].map(cabin0_mapping)
_input1['Cabin_2'].value_counts()
cabin2_mapping = {'S': 0, 'P': 1}
for dataset in train_test:
    dataset['Cabin_2'] = dataset['Cabin_2'].map(cabin2_mapping)
_input1.head()
_input0.head()
x = _input1.drop(['PassengerId', 'Transported'], axis=1)
y = _input1['Transported']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
params_clf = {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [4, 6, 8, 10, 12, 16], 'min_samples_leaf': [4, 8, 12, 16, 20], 'min_samples_split': [4, 8, 12, 16, 20]}
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(clf, param_distributions=params_clf, n_iter=30, scoring='roc_auc', n_jobs=-1)