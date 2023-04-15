import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
warnings.simplefilter(action='ignore')
train.head(80)
test.head()
train[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train.describe()
train.describe(include=['O'])
train.hist(figsize=(15, 10))

train_test = [train, test]
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
train.isnull().sum()
test.isnull().sum()
for dataset in train_test:
    dataset.drop('Name', axis=1, inplace=True)
train.head()
for dataset in train_test:
    dataset['CryoSleep'] = dataset['CryoSleep'].astype(int)
    dataset['VIP'] = dataset['VIP'].astype(int)
train.head()
test.head()
temp_Cabin_train = train['Cabin'].str.split('/', expand=True).rename(columns={0: 'Cabin_0', 1: 'Cabin_1', 2: 'Cabin_2'})
train = pd.concat([train, temp_Cabin_train], axis=1)
train.drop('Cabin', axis=1, inplace=True)
temp_Cabin_test = test['Cabin'].str.split('/', expand=True).rename(columns={0: 'Cabin_0', 1: 'Cabin_1', 2: 'Cabin_2'})
test = pd.concat([test, temp_Cabin_test], axis=1)
test.drop('Cabin', axis=1, inplace=True)
train.head()
test.head()
train_test = [train, test]
print('Age min', train['Age'].min())
print('Age max', train['Age'].max())
train['Age'].hist()

for dataset in train_test:
    dataset.loc[(dataset['Age'] >= 0) & (dataset['Age'] < 10), 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 10) & (dataset['Age'] < 20), 'Age'] = 10
    dataset.loc[(dataset['Age'] >= 20) & (dataset['Age'] < 30), 'Age'] = 20
    dataset.loc[(dataset['Age'] >= 30) & (dataset['Age'] < 40), 'Age'] = 30
    dataset.loc[(dataset['Age'] >= 40) & (dataset['Age'] < 50), 'Age'] = 40
    dataset.loc[(dataset['Age'] >= 50) & (dataset['Age'] < 60), 'Age'] = 50
    dataset.loc[(dataset['Age'] >= 60) & (dataset['Age'] < 70), 'Age'] = 60
    dataset.loc[dataset['Age'] >= 70, 'Age'] = 70
train.head()
plt.figure(figsize=[10, 10])
plt.subplot(2, 2, 1)
sns.countplot(data=train, x='HomePlanet', hue='Transported')
plt.title('HomePlanet vs. Transported')
plt.subplot(2, 2, 2)
sns.countplot(data=train, x='CryoSleep', hue='Transported')
plt.title('CryoSleep vs. Transported')
plt.subplot(2, 2, 3)
sns.countplot(data=train, x='Destination', hue='Transported')
plt.title('Destination vs. Transported')
plt.subplot(2, 2, 4)
sns.countplot(data=train, x='VIP', hue='Transported')
plt.title('VIP vs. Transported')
plt.subplots_adjust(left=0, right=1.5, wspace=0.3, hspace=0.3)


def bar_chart(feature):
    Transported = train[train['Transported'] == 1][feature].value_counts()
    Not = train[train['Transported'] == 0][feature].value_counts()
    df = pd.DataFrame([Transported, Not])
    df.index = ['Transported', 'Not']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))

def bar_non_stacked(feature):
    Transported = train[train['Transported'] == 1][feature].value_counts()
    Not = train[train['Transported'] == 0][feature].value_counts()
    df = pd.DataFrame([Transported, Not])
    df.index = ['Transported', 'Not']
    df.plot(kind='bar', stacked=False, figsize=(10, 5))
bar_chart('HomePlanet')
train[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('CryoSleep')
train[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Destination')
train[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Age')
bar_non_stacked('Age')
sns.countplot(data=train, x='Age', hue='Transported')
plt.title('Age vs. Transported')

train[['Age', 'Transported']].groupby(['Age'], as_index=False).mean()
bar_chart('VIP')
train[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train
bar_non_stacked('Cabin_0')
train[['Cabin_0', 'Transported']].groupby(['Cabin_0'], as_index=False).mean().sort_values(by='Transported', ascending=False)
bar_chart('Cabin_2')
train[['Cabin_2', 'Transported']].groupby(['Cabin_2'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train['HomePlanet'].value_counts()
planet_mapping = {'Earth': 0, 'Europa': 1, 'Mars': 2}
for dataset in train_test:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(planet_mapping)
train['Destination'].value_counts()
dest_mapping = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
for dataset in train_test:
    dataset['Destination'] = dataset['Destination'].map(dest_mapping)
train['Cabin_0'].value_counts()
cabin0_mapping = {'A': 0, 'B': 0.3, 'C': 0.6, 'D': 0.9, 'E': 1.2, 'F': 1.5, 'G': 1.8, 'T': 2.1}
for dataset in train_test:
    dataset['Cabin_0'] = dataset['Cabin_0'].map(cabin0_mapping)
train['Cabin_2'].value_counts()
cabin2_mapping = {'S': 0, 'P': 1}
for dataset in train_test:
    dataset['Cabin_2'] = dataset['Cabin_2'].map(cabin2_mapping)
train.head()
test.head()
x = train.drop(['PassengerId', 'Transported'], axis=1)
y = train['Transported']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
params_clf = {'n_estimators': [50, 100, 200, 300, 400], 'max_depth': [4, 6, 8, 10, 12, 16], 'min_samples_leaf': [4, 8, 12, 16, 20], 'min_samples_split': [4, 8, 12, 16, 20]}
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(clf, param_distributions=params_clf, n_iter=30, scoring='roc_auc', n_jobs=-1)