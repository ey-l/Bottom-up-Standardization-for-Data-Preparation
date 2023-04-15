import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.impute import SimpleImputer
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head(5)
train_data.info()
train_data.isnull().sum()
train_data[['Num_Passenger', 'G_Passenger']] = train_data['PassengerId'].str.split('_', expand=True).astype('int')
train_data.head(5)
train_data.drop(['PassengerId', 'Num_Passenger'], inplace=True, axis=1)
train_data.head(15)
train_data['Cabin'] = train_data['Cabin'].fillna('A/0/P')
train_data.isnull().sum()
train_data[['Deck', 'Num', 'Side']] = train_data.Cabin.str.split('/', expand=True)
train_data.head(5)
train_data.drop(['Cabin', 'Name'], axis=1, inplace=True)
train_data.head(5)
plt.figure(figsize=(10, 5))
sns.countplot(x='HomePlanet', data=train_data)
imp = SimpleImputer(strategy='most_frequent')
imp.fit_transform(train_data[['HomePlanet']])
train_data['HomePlanet'] = imp.fit_transform(train_data[['HomePlanet']])
plt.figure(figsize=(10, 6))
sns.countplot(x='CryoSleep', data=train_data)
imp = SimpleImputer(strategy='most_frequent')
train_data['CryoSleep'] = imp.fit_transform(train_data[['CryoSleep']])
plt.figure(figsize=(10, 5))
sns.countplot(x='Destination', data=train_data)
imp = SimpleImputer(strategy='most_frequent')
train_data['Destination'] = imp.fit_transform(train_data[['Destination']])
plt.figure(figsize=(10, 5))
sns.histplot(data=train_data, x='Age', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
imp = SimpleImputer(strategy='median')
train_data['Age'] = imp.fit_transform(train_data[['Age']])
plt.figure(figsize=(10, 5))
sns.countplot(x='VIP', data=train_data)
imp = SimpleImputer(strategy='most_frequent')
train_data['VIP'] = imp.fit_transform(train_data[['VIP']])
train_data.head()
imp = SimpleImputer(strategy='constant', fill_value=0)
train_data['RoomService'] = imp.fit_transform(train_data[['RoomService']])
imp = SimpleImputer(strategy='constant', fill_value=0)
train_data['FoodCourt'] = imp.fit_transform(train_data[['FoodCourt']])
imp = SimpleImputer(strategy='constant', fill_value=0)
train_data['ShoppingMall'] = imp.fit_transform(train_data[['ShoppingMall']])
imp = SimpleImputer(strategy='constant', fill_value=0)
train_data['Spa'] = imp.fit_transform(train_data[['Spa']])
imp = SimpleImputer(strategy='constant', fill_value=0)
train_data['VRDeck'] = imp.fit_transform(train_data[['VRDeck']])
train_data.isnull().sum()
train_data.info()
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Num']
for i in categorical_cols:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate([train_data[i]], axis=0).astype(str)