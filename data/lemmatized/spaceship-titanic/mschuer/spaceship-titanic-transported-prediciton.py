import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
cf.go_offline()
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.isnull().sum()
_input0.head()
_input0.isnull().sum()
_input0['Cabin']
_input0.info()
_input1.describe()
_input0.describe()
len(_input1)
len(_input0)
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='CryoSleep', hue='HomePlanet')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='CryoSleep', hue='HomePlanet')
_input1['HomePlanet'].value_counts()
_input0['HomePlanet'].value_counts()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Transported', hue='HomePlanet')
plt.figure(figsize=(20, 2))
sns.countplot(data=_input1, y='Transported')
_input1['Transported'].value_counts() / len(_input1['Transported'])
_input1['CryoSleep'].value_counts() / len(_input1['CryoSleep'])
_input0['CryoSleep'].value_counts() / len(_input1['CryoSleep'])
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.corr(), annot=True)
_input1.corr()['Transported'].sort_values()[:-1].plot(kind='bar')
_input1['Age'].isnull().sum()
_input0['Age'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.boxplot(data=_input0, x='HomePlanet', y='Age')
plt.title('Age data given HomePlanet for test data set')
plt.figure(figsize=(14, 8))
sns.boxplot(data=_input1, x='HomePlanet', y='Age')
plt.title('Age data given HomePlanet for training data set')
_input1['Age'].groupby(_input1['HomePlanet']).median()
_input0['Age'].groupby(_input0['HomePlanet']).median()

def impute_age(cols):
    Age = cols[0]
    HomePlanet = cols[1]
    if pd.isnull(Age):
        if HomePlanet == 'Earth':
            return 23
        elif HomePlanet == 'Europa':
            return 33
        elif HomePlanet == 'Mars':
            return 28
        else:
            return _input1['Age'].median()
    else:
        return Age

def impute_age_test_set(cols):
    Age = cols[0]
    HomePlanet = cols[1]
    if pd.isnull(Age):
        if HomePlanet == 'Earth':
            return 23
        elif HomePlanet == 'Europa':
            return 32
        elif HomePlanet == 'Mars':
            return 27
        else:
            return _input0['Age'].median()
    else:
        return Age
_input1['Age'] = _input1[['Age', 'HomePlanet']].apply(impute_age, axis=1)
_input0['Age'] = _input0[['Age', 'HomePlanet']].apply(impute_age_test_set, axis=1)
_input0.isnull().sum()
_input1.isnull().sum()
_input1['HomePlanet'].isnull().sum()
_input1['Destination'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Destination', hue='HomePlanet')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Destination', hue='Transported')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Destination', hue='CryoSleep')
_input1['Cabin'].mode()
_input0['Cabin'].mode()
_input1['Cabin'] = _input1['Cabin'].replace(np.nan, 'G/734/S')
_input0['Cabin'] = _input0['Cabin'].replace(np.nan, 'G/160/P')
_input0['Cabin'].isnull().sum()
_input1['Cabin'].isnull().sum()

def cabin_deck(cols):
    Cabin = cols[0]
    if pd.isnull(Cabin):
        return Cabin.replace(np.nan, 'G/734/S')
    else:
        return Cabin.split('/')[0].strip()

def cabin_deck_test_set(cols):
    Cabin = cols[0]
    if pd.isnull(Cabin):
        return Cabin.replace(np.nan, 'G/160/P')
    else:
        return Cabin.split('/')[0].strip()
_input1['Cabin_deck'] = _input1['Cabin'].apply(cabin_deck)
_input0['Cabin_deck'] = _input0['Cabin'].apply(cabin_deck_test_set)
_input1['Cabin_deck'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x=_input1['Cabin_deck'].sort_values(), hue='Transported')
_input1[_input1['Cabin_deck'].isin(['F', 'G'])]
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x=_input1['Cabin_deck'].sort_values(), hue='Destination')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x=_input0['Cabin_deck'].sort_values(), hue='Destination')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x=_input1['Cabin_deck'].sort_values(), hue='CryoSleep')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x=_input0['Cabin_deck'].sort_values(), hue='CryoSleep')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
_input1['Cabin_side'] = [x.split('/')[2].strip() for x in _input1['Cabin']]
_input0['Cabin_side'] = [x.split('/')[2].strip() for x in _input0['Cabin']]
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Cabin_side', hue='Transported')
plt.title('Transported by side of ship')
_input1['Transported'].groupby(_input1['Cabin_side']).mean()
_input1 = _input1.drop('Name', axis=1)
_input0 = _input0.drop('Name', axis=1)
plt.figure(figsize=(14, 8))
sns.displot(_input1['Age'], kde=False, bins=50)
_input1['age_band'] = pd.cut(_input1['Age'], 4)
_input0['age_band'] = pd.cut(_input0['Age'], 4)
_input1['age_band'].value_counts()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='age_band', hue='Transported')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='age_band', hue='Destination')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='age_band', hue='Destination')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='age_band', hue='HomePlanet')
_input1['HomePlanet'].value_counts() / len(_input1)
_input0['HomePlanet'].value_counts() / len(_input0)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='age_band', hue='CryoSleep')
_input1.loc[(_input1['Age'] > 19.75) & (_input1['Age'] <= 39.5) & (_input1['Destination'] == 'TRAPPIST-1e')]
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='CryoSleep', hue='HomePlanet')
_input1['CryoSleep'].isnull().sum()
_input1['CryoSleep'].value_counts() / len(_input1)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(np.random.choice(_input1['CryoSleep']))
_input1['CryoSleep'].value_counts() / len(_input1)
_input1['CryoSleep'].isnull().sum()
_input0['CryoSleep'].value_counts() / len(_input0)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(np.random.choice(_input0['CryoSleep']))
_input0['CryoSleep'].value_counts() / len(_input0)
_input1['CryoSleep'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='age_band', hue='CryoSleep')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='VIP', hue='Transported')
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.corr(), annot=True)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
plt.figure(figsize=(14, 8))
sns.histplot(_input1['RoomService'], bins=35)
plt.figure(figsize=(14, 8))
sns.histplot(_input1['FoodCourt'], bins=35)
_input1['FoodCourt'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(_input1['ShoppingMall'], bins=35)
_input1['ShoppingMall'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(_input1['Spa'], bins=35)
_input1['Spa'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(_input1['VRDeck'], bins=35)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['ShoppingMall'], data=_input0)
_input1[_input1['ShoppingMall'] > 12000]
_input1 = _input1.drop(index=[6223, 8415])
_input1['ShoppingMall'].mean()
_input0['ShoppingMall'].mean()
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input1['FoodCourt'], data=_input1)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['FoodCourt'], data=_input0)
_input1[_input1['FoodCourt'] > 20000]
_input0[_input0['FoodCourt'] > 15000]
_input1 = _input1.drop(index=[1213, 1842, 2067, 3198, 3538])
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['FoodCourt'], data=_input0)
_input1.isnull().sum()
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input1['RoomService'], data=_input1)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['RoomService'], data=_input0)
_input1[_input1['RoomService'] > 10000]
_input0[_input0['RoomService'] > 8000]
_input1 = _input1.drop(index=[4416])
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['RoomService'], data=_input0)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input1['Spa'], data=_input1)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['Spa'], data=_input0)
_input1[_input1['Spa'] > 15000]
_input0[_input0['Spa'] > 12000]
_input1 = _input1.drop(index=[1095, 1390, 1598, 4278, 5722, 6921, 7995])
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['Spa'], data=_input0)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input1['VRDeck'], data=_input1)
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['VRDeck'], data=_input0)
_input1[_input1['VRDeck'] > 15000]
_input0[_input0['VRDeck'] > 15000]
_input1 = _input1.drop(index=[725, 3366, 4311, 5619, 6547])
plt.figure(figsize=(14, 8))
sns.boxplot(x=_input0['VRDeck'], data=_input0)
sns.boxplot(data=_input1, x='VRDeck')
_input1 = _input1.drop('VRDeck', axis=1)
_input0 = _input0.drop('VRDeck', axis=1)
for i in _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']]:
    print(i, _input1[_input1[i] > 0].mean()[i])
for i in _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']]:
    print(i, _input0[_input0[i] > 0].mean()[i])
_input1['RoomService'] = _input1['RoomService'].fillna(648)
_input0['RoomService'] = _input0['RoomService'].fillna(626)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(1226)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(1238)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(497)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(516)
_input1['Spa'] = _input1['Spa'].fillna(796)
_input0['Spa'] = _input0['Spa'].fillna(808)
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input0.isnull().sum()
plt.figure(figsize=(14, 8))
sns.histplot(_input1['Spa'], bins=35)
_input1['RoomService'] = pd.cut(_input1['RoomService'], bins=[-1, 2000, 6000, 10000], labels=['low', 'med', 'high'])
_input1['RoomService'] = _input1['RoomService'].replace({'low': 0, 'med': 1, 'high': 2})
_input0['RoomService'] = pd.cut(_input0['RoomService'], bins=[-1, 2000, 6000, 20000], labels=['low', 'med', 'high'])
_input0['RoomService'] = _input0['RoomService'].replace({'low': 0, 'med': 1, 'high': 2})
_input1['FoodCourt'] = pd.cut(_input1['FoodCourt'], bins=[-1, 6000, 12000, 26000], labels=['low', 'med', 'high'])
_input1['FoodCourt'] = _input1['FoodCourt'].replace({'low': 0, 'med': 1, 'high': 2})
_input0['FoodCourt'] = pd.cut(_input0['FoodCourt'], bins=[-1, 6000, 12000, 27000], labels=['low', 'med', 'high'])
_input0['FoodCourt'] = _input0['FoodCourt'].replace({'low': 0, 'med': 1, 'high': 2})
_input1['ShoppingMall'] = pd.cut(_input1['ShoppingMall'], bins=[-1, 3000, 7000, 12000], labels=['low', 'med', 'high'])
_input1['ShoppingMall'] = _input1['ShoppingMall'].replace({'low': 0, 'med': 1, 'high': 2})
_input0['ShoppingMall'] = pd.cut(_input0['ShoppingMall'], bins=[-1, 3000, 7000, 20000], labels=['low', 'med', 'high'])
_input0['ShoppingMall'] = _input0['ShoppingMall'].replace({'low': 0, 'med': 1, 'high': 2})
_input1['Spa'] = pd.cut(_input1['Spa'], bins=[-1, 5000, 10000, 15000], labels=['low', 'med', 'high'])
_input1['Spa'] = _input1['Spa'].replace({'low': 0, 'med': 1, 'high': 2})
_input0['Spa'] = pd.cut(_input0['Spa'], bins=[-1, 5000, 10000, 25000], labels=['low', 'med', 'high'])
_input0['Spa'] = _input0['Spa'].replace({'low': 0, 'med': 1, 'high': 2})
len(_input0)
_input1['VIP'].isnull().sum() / len(_input1)
_input0['VIP'].isnull().sum() / len(_input0)
_input1['VIP'].value_counts()
_input0['VIP'].value_counts()
_input1['VIP'] = _input1['VIP'].fillna(np.random.choice(_input1['VIP'][~_input1['VIP'].isna()]))
_input0['VIP'] = _input0['VIP'].fillna(np.random.choice(_input0['VIP'][~_input0['VIP'].isna()]))
_input1['VIP'].value_counts()
_input0['VIP'].value_counts()
_input1.isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input1['HomePlanet'].isnull().sum() / len(_input1)
_input0['HomePlanet'].isnull().sum() / len(_input0)
_input0.head()
_input0 = _input0.drop('Cabin', axis=1)
_input1.head()
_input0.head()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='CryoSleep', hue='HomePlanet')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='CryoSleep', hue='HomePlanet')
_input1[_input1['HomePlanet'] == 'Europa']['HomePlanet'].value_counts() / len(_input1)
_input0[_input0['HomePlanet'] == 'Europa']['HomePlanet'].value_counts() / len(_input0)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Destination', hue='HomePlanet')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='Destination', hue='HomePlanet')

def home_planet(col):
    Home = col[0]
    Destination = col[1]
    if pd.isnull(Home):
        if Destination == '55 Cancri e':
            return 'Europa'
        elif Destination == 'TRAPPIST-1e ':
            return 'Mars'
        else:
            return 'Earth'
    else:
        return Home
_input1['HomePlanet'] = _input1[['HomePlanet', 'Destination']].apply(home_planet, axis=1)
_input1['HomePlanet'].value_counts() / len(_input1)
_input1.isnull().sum()
_input0['HomePlanet'].value_counts() / len(_input0)

def home_planet_test_set(col):
    Home = col[0]
    Destination = col[1]
    if pd.isnull(Home):
        if Destination == '55 Cancri e':
            return 'Europa'
        elif Destination == 'TRAPPIST-1e ':
            return 'Mars'
        else:
            return 'Earth'
    else:
        return Home
_input0['HomePlanet'] = _input0[['HomePlanet', 'Destination']].apply(home_planet_test_set, axis=1)
_input0['HomePlanet'].value_counts() / len(_input0)
_input0.isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input1['Destination'].isnull().sum() / len(_input1)
_input0['Destination'].isnull().sum() / len(_input0)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Destination', hue='HomePlanet')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='Destination', hue='HomePlanet')
_input1['HomePlanet'].value_counts() / len(_input1)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='HomePlanet', hue='Destination')
_input1.isnull().sum()

def dest_impute(col):
    Destination = col[0]
    HomePlanet = col[1]
    if pd.isnull(Destination):
        if HomePlanet == 'Europa':
            return '55 Cancri e'
        elif HomePlanet == 'Mars':
            return 'TRAPPIST-1e'
        else:
            return 'TRAPPIST-1e'
    else:
        return Destination
_input1['Destination'].value_counts()
_input1['Destination'] = _input1[['Destination', 'HomePlanet']].apply(dest_impute, axis=1)
_input1.isnull().sum()

def dest_impute_test_set(col):
    Destination = col[0]
    HomePlanet = col[1]
    if pd.isnull(Destination):
        if HomePlanet == 'Europa':
            return '55 Cancri e'
        elif HomePlanet == 'Mars':
            return 'TRAPPIST-1e'
        else:
            return 'TRAPPIST-1e'
    else:
        return Destination
_input0['Destination'] = _input0[['Destination', 'HomePlanet']].apply(dest_impute_test_set, axis=1)
_input0.isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='HomePlanet', hue='Destination')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='HomePlanet', hue='Destination')
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input1['age_band'].value_counts()
_input0['age_band'].value_counts()
_input1.loc[_input1['Age'] <= 19.75, 'Age'] = 0
_input1.loc[(_input1['Age'] > 19.75) & (_input1['Age'] <= 39.5), 'Age'] = 1
_input1.loc[(_input1['Age'] > 39.5) & (_input1['Age'] <= 59.25), 'Age'] = 2
_input1.loc[(_input1['Age'] > 59.25) & (_input1['Age'] <= 79), 'Age'] = 3
_input1.loc[_input1['Age'] > 79, 'Age'] = 4
_input1.head()
_input0.loc[_input0['Age'] <= 19.75, 'Age'] = 0
_input0.loc[(_input0['Age'] > 19.75) & (_input0['Age'] <= 39.5), 'Age'] = 1
_input0.loc[(_input0['Age'] > 39.5) & (_input0['Age'] <= 59.25), 'Age'] = 2
_input0.loc[(_input0['Age'] > 59.25) & (_input0['Age'] <= 79), 'Age'] = 3
_input0.loc[_input0['Age'] > 79, 'Age'] = 4
_input1.isnull().sum()
_input1 = _input1.drop('age_band', axis=1)
_input0 = _input0.drop('age_band', axis=1)
_input0.isnull().sum()
_input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1})
_input0['VIP'] = _input0['VIP'].replace({False: 0, True: 1})
_input1['HomePlanet'] = _input1['HomePlanet'].replace({'Earth': 0, 'Mars': 1, 'Europa': 2})
_input0['HomePlanet'] = _input0['HomePlanet'].replace({'Earth': 0, 'Mars': 1, 'Europa': 2})
_input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1})
_input0['CryoSleep'] = _input0['CryoSleep'].replace({False: 0, True: 1})
_input1['Destination'] = _input1['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
_input0['Destination'] = _input0['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
_input0.isnull().sum()
_input1['Transported'] = _input1['Transported'].replace({False: 0, True: 1})
cabin_deck = pd.get_dummies(_input1['Cabin_deck'], drop_first=True)
cabin_side = pd.get_dummies(_input1['Cabin_side'], drop_first=True)
_input1 = pd.concat([_input1, cabin_deck, cabin_side], axis=1)
_input1 = _input1.drop(['Cabin_deck', 'Cabin_side'], axis=1)
cabin_deck_test = pd.get_dummies(_input0['Cabin_deck'], drop_first=True)
cabin_side_test = pd.get_dummies(_input0['Cabin_side'], drop_first=True)
_input0 = pd.concat([_input0, cabin_deck_test, cabin_side_test], axis=1)
_input0 = _input0.drop(['Cabin_deck', 'Cabin_side'], axis=1)
_input1.info()
_input1 = _input1.drop('PassengerId', axis=1)
_input1.isnull().sum()
_input1 = _input1.drop('Cabin', axis=1)
_input1.isnull().sum()
X_train = _input1.drop('Transported', axis=1)
y_train = _input1['Transported']
X_test = _input0.drop('PassengerId', axis=1)
logmodel = LogisticRegression()