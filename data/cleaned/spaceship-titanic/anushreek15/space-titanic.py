import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
train.isnull().sum()
column = train[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
for i in column:
    print(column[i].value_counts())
column = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in column:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
    print(col)
sns.distplot(x=train['Age'])
sns.distplot(x=test['Age'])
column = ['Age']
for i in column:
    train[i].fillna(train[i].median(), inplace=True)
    test[i].fillna(test[i].median(), inplace=True)
    print(i)
Spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in Spendings:
    train[i].fillna(train[i].median(), inplace=True)
    test[i].fillna(test[i].median(), inplace=True)
    print(i)
train = train.drop(columns=['Cabin', 'Name'], axis=1)
test = test.drop(columns=['Cabin', 'Name'], axis=1)
train.head()
test.head()
train[['CryoSleep', 'VIP']] = train[['CryoSleep', 'VIP']].astype('int')
test[['CryoSleep', 'VIP']] = test[['CryoSleep', 'VIP']].astype('int')
train.head()
test.head()
cat_var = train[['HomePlanet', 'Destination']]
num_val = pd.get_dummies(cat_var)
cat_var_test = test[['HomePlanet', 'Destination']]
num_val_test = pd.get_dummies(cat_var_test)
num_val_test.head()
num_val.head()
train = train.drop(columns=['HomePlanet', 'Destination'])
test = test.drop(columns=['HomePlanet', 'Destination'])
train = pd.concat([train, num_val], axis=1)
test = pd.concat([test, num_val_test], axis=1)
X = train.drop(columns=['Transported'], axis=1)
y = train['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(max_depth=9, random_state=0)