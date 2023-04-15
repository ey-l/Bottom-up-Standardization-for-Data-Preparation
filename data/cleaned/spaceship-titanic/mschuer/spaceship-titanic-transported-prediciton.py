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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
gender_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
train.isnull().sum()
test.head()
test.isnull().sum()
test['Cabin']
test.info()
train.describe()
test.describe()
len(train)
len(test)
plt.figure(figsize=(14, 8))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(test.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='CryoSleep', hue='HomePlanet')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='CryoSleep', hue='HomePlanet')

train['HomePlanet'].value_counts()
test['HomePlanet'].value_counts()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Transported', hue='HomePlanet')

plt.figure(figsize=(20, 2))
sns.countplot(data=train, y='Transported')

train['Transported'].value_counts() / len(train['Transported'])
train['CryoSleep'].value_counts() / len(train['CryoSleep'])
test['CryoSleep'].value_counts() / len(train['CryoSleep'])
plt.figure(figsize=(14, 8))
sns.heatmap(train.corr(), annot=True)

train.corr()['Transported'].sort_values()[:-1].plot(kind='bar')
train['Age'].isnull().sum()
test['Age'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.boxplot(data=test, x='HomePlanet', y='Age')
plt.title('Age data given HomePlanet for test data set')

plt.figure(figsize=(14, 8))
sns.boxplot(data=train, x='HomePlanet', y='Age')
plt.title('Age data given HomePlanet for training data set')

train['Age'].groupby(train['HomePlanet']).median()
test['Age'].groupby(test['HomePlanet']).median()

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
            return train['Age'].median()
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
            return test['Age'].median()
    else:
        return Age
train['Age'] = train[['Age', 'HomePlanet']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'HomePlanet']].apply(impute_age_test_set, axis=1)
test.isnull().sum()
train.isnull().sum()
train['HomePlanet'].isnull().sum()
train['Destination'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Destination', hue='HomePlanet')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Destination', hue='Transported')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Destination', hue='CryoSleep')

train['Cabin'].mode()
test['Cabin'].mode()
train['Cabin'] = train['Cabin'].replace(np.nan, 'G/734/S')
test['Cabin'] = test['Cabin'].replace(np.nan, 'G/160/P')
test['Cabin'].isnull().sum()
train['Cabin'].isnull().sum()

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
train['Cabin_deck'] = train['Cabin'].apply(cabin_deck)
test['Cabin_deck'] = test['Cabin'].apply(cabin_deck_test_set)
train['Cabin_deck'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x=train['Cabin_deck'].sort_values(), hue='Transported')

train[train['Cabin_deck'].isin(['F', 'G'])]
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x=train['Cabin_deck'].sort_values(), hue='Destination')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x=test['Cabin_deck'].sort_values(), hue='Destination')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x=train['Cabin_deck'].sort_values(), hue='CryoSleep')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x=test['Cabin_deck'].sort_values(), hue='CryoSleep')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='CryoSleep', hue='Transported')

train['Cabin_side'] = [x.split('/')[2].strip() for x in train['Cabin']]
test['Cabin_side'] = [x.split('/')[2].strip() for x in test['Cabin']]
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Cabin_side', hue='Transported')
plt.title('Transported by side of ship')

train['Transported'].groupby(train['Cabin_side']).mean()
train = train.drop('Name', axis=1)
test = test.drop('Name', axis=1)
plt.figure(figsize=(14, 8))
sns.displot(train['Age'], kde=False, bins=50)

train['age_band'] = pd.cut(train['Age'], 4)
test['age_band'] = pd.cut(test['Age'], 4)
train['age_band'].value_counts()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='age_band', hue='Transported')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='age_band', hue='Destination')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='age_band', hue='Destination')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='age_band', hue='HomePlanet')

train['HomePlanet'].value_counts() / len(train)
test['HomePlanet'].value_counts() / len(test)
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='age_band', hue='CryoSleep')

train.loc[(train['Age'] > 19.75) & (train['Age'] <= 39.5) & (train['Destination'] == 'TRAPPIST-1e')]
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='CryoSleep', hue='HomePlanet')

train['CryoSleep'].isnull().sum()
train['CryoSleep'].value_counts() / len(train)
train['CryoSleep'] = train['CryoSleep'].fillna(np.random.choice(train['CryoSleep']))
train['CryoSleep'].value_counts() / len(train)
train['CryoSleep'].isnull().sum()
test['CryoSleep'].value_counts() / len(test)
test['CryoSleep'] = test['CryoSleep'].fillna(np.random.choice(test['CryoSleep']))
test['CryoSleep'].value_counts() / len(test)
train['CryoSleep'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='age_band', hue='CryoSleep')

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='VIP', hue='Transported')

plt.figure(figsize=(14, 8))
sns.heatmap(train.corr(), annot=True)

plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='CryoSleep', hue='Transported')

plt.figure(figsize=(14, 8))
sns.histplot(train['RoomService'], bins=35)

plt.figure(figsize=(14, 8))
sns.histplot(train['FoodCourt'], bins=35)

train['FoodCourt'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(train['ShoppingMall'], bins=35)

train['ShoppingMall'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(train['Spa'], bins=35)

train['Spa'].describe()
plt.figure(figsize=(14, 8))
sns.histplot(train['VRDeck'], bins=35)

plt.figure(figsize=(14, 8))
sns.boxplot(x=test['ShoppingMall'], data=test)

train[train['ShoppingMall'] > 12000]
train = train.drop(index=[6223, 8415])
train['ShoppingMall'].mean()
test['ShoppingMall'].mean()
plt.figure(figsize=(14, 8))
sns.boxplot(x=train['FoodCourt'], data=train)

plt.figure(figsize=(14, 8))
sns.boxplot(x=test['FoodCourt'], data=test)

train[train['FoodCourt'] > 20000]
test[test['FoodCourt'] > 15000]
train = train.drop(index=[1213, 1842, 2067, 3198, 3538])
plt.figure(figsize=(14, 8))
sns.boxplot(x=test['FoodCourt'], data=test)

train.isnull().sum()
plt.figure(figsize=(14, 8))
sns.boxplot(x=train['RoomService'], data=train)

plt.figure(figsize=(14, 8))
sns.boxplot(x=test['RoomService'], data=test)

train[train['RoomService'] > 10000]
test[test['RoomService'] > 8000]
train = train.drop(index=[4416])
plt.figure(figsize=(14, 8))
sns.boxplot(x=test['RoomService'], data=test)

plt.figure(figsize=(14, 8))
sns.boxplot(x=train['Spa'], data=train)

plt.figure(figsize=(14, 8))
sns.boxplot(x=test['Spa'], data=test)

train[train['Spa'] > 15000]
test[test['Spa'] > 12000]
train = train.drop(index=[1095, 1390, 1598, 4278, 5722, 6921, 7995])
plt.figure(figsize=(14, 8))
sns.boxplot(x=test['Spa'], data=test)

plt.figure(figsize=(14, 8))
sns.boxplot(x=train['VRDeck'], data=train)

plt.figure(figsize=(14, 8))
sns.boxplot(x=test['VRDeck'], data=test)

train[train['VRDeck'] > 15000]
test[test['VRDeck'] > 15000]
train = train.drop(index=[725, 3366, 4311, 5619, 6547])
plt.figure(figsize=(14, 8))
sns.boxplot(x=test['VRDeck'], data=test)

sns.boxplot(data=train, x='VRDeck')
train = train.drop('VRDeck', axis=1)
test = test.drop('VRDeck', axis=1)
for i in train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']]:
    print(i, train[train[i] > 0].mean()[i])
for i in test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']]:
    print(i, test[test[i] > 0].mean()[i])
train['RoomService'] = train['RoomService'].fillna(648)
test['RoomService'] = test['RoomService'].fillna(626)
train['FoodCourt'] = train['FoodCourt'].fillna(1226)
test['FoodCourt'] = test['FoodCourt'].fillna(1238)
train['ShoppingMall'] = train['ShoppingMall'].fillna(497)
test['ShoppingMall'] = test['ShoppingMall'].fillna(516)
train['Spa'] = train['Spa'].fillna(796)
test['Spa'] = test['Spa'].fillna(808)
plt.figure(figsize=(14, 8))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test.isnull(), yticklabels=False, cmap='viridis')

test.isnull().sum()
plt.figure(figsize=(14, 8))
sns.histplot(train['Spa'], bins=35)

train['RoomService'] = pd.cut(train['RoomService'], bins=[-1, 2000, 6000, 10000], labels=['low', 'med', 'high'])
train['RoomService'] = train['RoomService'].replace({'low': 0, 'med': 1, 'high': 2})
test['RoomService'] = pd.cut(test['RoomService'], bins=[-1, 2000, 6000, 20000], labels=['low', 'med', 'high'])
test['RoomService'] = test['RoomService'].replace({'low': 0, 'med': 1, 'high': 2})
train['FoodCourt'] = pd.cut(train['FoodCourt'], bins=[-1, 6000, 12000, 26000], labels=['low', 'med', 'high'])
train['FoodCourt'] = train['FoodCourt'].replace({'low': 0, 'med': 1, 'high': 2})
test['FoodCourt'] = pd.cut(test['FoodCourt'], bins=[-1, 6000, 12000, 27000], labels=['low', 'med', 'high'])
test['FoodCourt'] = test['FoodCourt'].replace({'low': 0, 'med': 1, 'high': 2})
train['ShoppingMall'] = pd.cut(train['ShoppingMall'], bins=[-1, 3000, 7000, 12000], labels=['low', 'med', 'high'])
train['ShoppingMall'] = train['ShoppingMall'].replace({'low': 0, 'med': 1, 'high': 2})
test['ShoppingMall'] = pd.cut(test['ShoppingMall'], bins=[-1, 3000, 7000, 20000], labels=['low', 'med', 'high'])
test['ShoppingMall'] = test['ShoppingMall'].replace({'low': 0, 'med': 1, 'high': 2})
train['Spa'] = pd.cut(train['Spa'], bins=[-1, 5000, 10000, 15000], labels=['low', 'med', 'high'])
train['Spa'] = train['Spa'].replace({'low': 0, 'med': 1, 'high': 2})
test['Spa'] = pd.cut(test['Spa'], bins=[-1, 5000, 10000, 25000], labels=['low', 'med', 'high'])
test['Spa'] = test['Spa'].replace({'low': 0, 'med': 1, 'high': 2})
len(test)
train['VIP'].isnull().sum() / len(train)
test['VIP'].isnull().sum() / len(test)
train['VIP'].value_counts()
test['VIP'].value_counts()
train['VIP'] = train['VIP'].fillna(np.random.choice(train['VIP'][~train['VIP'].isna()]))
test['VIP'] = test['VIP'].fillna(np.random.choice(test['VIP'][~test['VIP'].isna()]))
train['VIP'].value_counts()
test['VIP'].value_counts()
train.isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test.isnull(), yticklabels=False, cmap='viridis')

train['HomePlanet'].isnull().sum() / len(train)
test['HomePlanet'].isnull().sum() / len(test)
test.head()
test = test.drop('Cabin', axis=1)
train.head()
test.head()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='CryoSleep', hue='HomePlanet')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='CryoSleep', hue='HomePlanet')

train[train['HomePlanet'] == 'Europa']['HomePlanet'].value_counts() / len(train)
test[test['HomePlanet'] == 'Europa']['HomePlanet'].value_counts() / len(test)
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Destination', hue='HomePlanet')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='Destination', hue='HomePlanet')


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
train['HomePlanet'] = train[['HomePlanet', 'Destination']].apply(home_planet, axis=1)
train['HomePlanet'].value_counts() / len(train)
train.isnull().sum()
test['HomePlanet'].value_counts() / len(test)

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
test['HomePlanet'] = test[['HomePlanet', 'Destination']].apply(home_planet_test_set, axis=1)
test['HomePlanet'].value_counts() / len(test)
test.isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(test.isnull(), yticklabels=False, cmap='viridis')

train['Destination'].isnull().sum() / len(train)
test['Destination'].isnull().sum() / len(test)
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='Destination', hue='HomePlanet')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='Destination', hue='HomePlanet')

train['HomePlanet'].value_counts() / len(train)
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='HomePlanet', hue='Destination')

train.isnull().sum()

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
train['Destination'].value_counts()
train['Destination'] = train[['Destination', 'HomePlanet']].apply(dest_impute, axis=1)
train.isnull().sum()

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
test['Destination'] = test[['Destination', 'HomePlanet']].apply(dest_impute_test_set, axis=1)
test.isnull().sum()
plt.figure(figsize=(14, 8))
sns.countplot(data=train, x='HomePlanet', hue='Destination')

plt.figure(figsize=(14, 8))
sns.countplot(data=test, x='HomePlanet', hue='Destination')

plt.figure(figsize=(14, 8))
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test.isnull(), yticklabels=False, cmap='viridis')

train['age_band'].value_counts()
test['age_band'].value_counts()
train.loc[train['Age'] <= 19.75, 'Age'] = 0
train.loc[(train['Age'] > 19.75) & (train['Age'] <= 39.5), 'Age'] = 1
train.loc[(train['Age'] > 39.5) & (train['Age'] <= 59.25), 'Age'] = 2
train.loc[(train['Age'] > 59.25) & (train['Age'] <= 79), 'Age'] = 3
train.loc[train['Age'] > 79, 'Age'] = 4
train.head()
test.loc[test['Age'] <= 19.75, 'Age'] = 0
test.loc[(test['Age'] > 19.75) & (test['Age'] <= 39.5), 'Age'] = 1
test.loc[(test['Age'] > 39.5) & (test['Age'] <= 59.25), 'Age'] = 2
test.loc[(test['Age'] > 59.25) & (test['Age'] <= 79), 'Age'] = 3
test.loc[test['Age'] > 79, 'Age'] = 4
train.isnull().sum()
train = train.drop('age_band', axis=1)
test = test.drop('age_band', axis=1)
test.isnull().sum()
train['VIP'] = train['VIP'].replace({False: 0, True: 1})
test['VIP'] = test['VIP'].replace({False: 0, True: 1})
train['HomePlanet'] = train['HomePlanet'].replace({'Earth': 0, 'Mars': 1, 'Europa': 2})
test['HomePlanet'] = test['HomePlanet'].replace({'Earth': 0, 'Mars': 1, 'Europa': 2})
train['CryoSleep'] = train['CryoSleep'].replace({False: 0, True: 1})
test['CryoSleep'] = test['CryoSleep'].replace({False: 0, True: 1})
train['Destination'] = train['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
test['Destination'] = test['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2})
test.isnull().sum()
train['Transported'] = train['Transported'].replace({False: 0, True: 1})
cabin_deck = pd.get_dummies(train['Cabin_deck'], drop_first=True)
cabin_side = pd.get_dummies(train['Cabin_side'], drop_first=True)
train = pd.concat([train, cabin_deck, cabin_side], axis=1)
train = train.drop(['Cabin_deck', 'Cabin_side'], axis=1)
cabin_deck_test = pd.get_dummies(test['Cabin_deck'], drop_first=True)
cabin_side_test = pd.get_dummies(test['Cabin_side'], drop_first=True)
test = pd.concat([test, cabin_deck_test, cabin_side_test], axis=1)
test = test.drop(['Cabin_deck', 'Cabin_side'], axis=1)
train.info()
train = train.drop('PassengerId', axis=1)
train.isnull().sum()
train = train.drop('Cabin', axis=1)
train.isnull().sum()
X_train = train.drop('Transported', axis=1)
y_train = train['Transported']
X_test = test.drop('PassengerId', axis=1)
logmodel = LogisticRegression()