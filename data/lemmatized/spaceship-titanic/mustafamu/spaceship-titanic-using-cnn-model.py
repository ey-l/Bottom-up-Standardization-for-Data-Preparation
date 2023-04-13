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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(5)
_input1.info()
_input1.isnull().sum()
_input1[['Num_Passenger', 'G_Passenger']] = _input1['PassengerId'].str.split('_', expand=True).astype('int')
_input1.head(5)
_input1 = _input1.drop(['PassengerId', 'Num_Passenger'], inplace=False, axis=1)
_input1.head(15)
_input1['Cabin'] = _input1['Cabin'].fillna('A/0/P')
_input1.isnull().sum()
_input1[['Deck', 'Num', 'Side']] = _input1.Cabin.str.split('/', expand=True)
_input1.head(5)
_input1 = _input1.drop(['Cabin', 'Name'], axis=1, inplace=False)
_input1.head(5)
plt.figure(figsize=(10, 5))
sns.countplot(x='HomePlanet', data=_input1)
imp = SimpleImputer(strategy='most_frequent')
imp.fit_transform(_input1[['HomePlanet']])
_input1['HomePlanet'] = imp.fit_transform(_input1[['HomePlanet']])
plt.figure(figsize=(10, 6))
sns.countplot(x='CryoSleep', data=_input1)
imp = SimpleImputer(strategy='most_frequent')
_input1['CryoSleep'] = imp.fit_transform(_input1[['CryoSleep']])
plt.figure(figsize=(10, 5))
sns.countplot(x='Destination', data=_input1)
imp = SimpleImputer(strategy='most_frequent')
_input1['Destination'] = imp.fit_transform(_input1[['Destination']])
plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='Age', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
imp = SimpleImputer(strategy='median')
_input1['Age'] = imp.fit_transform(_input1[['Age']])
plt.figure(figsize=(10, 5))
sns.countplot(x='VIP', data=_input1)
imp = SimpleImputer(strategy='most_frequent')
_input1['VIP'] = imp.fit_transform(_input1[['VIP']])
_input1.head()
imp = SimpleImputer(strategy='constant', fill_value=0)
_input1['RoomService'] = imp.fit_transform(_input1[['RoomService']])
imp = SimpleImputer(strategy='constant', fill_value=0)
_input1['FoodCourt'] = imp.fit_transform(_input1[['FoodCourt']])
imp = SimpleImputer(strategy='constant', fill_value=0)
_input1['ShoppingMall'] = imp.fit_transform(_input1[['ShoppingMall']])
imp = SimpleImputer(strategy='constant', fill_value=0)
_input1['Spa'] = imp.fit_transform(_input1[['Spa']])
imp = SimpleImputer(strategy='constant', fill_value=0)
_input1['VRDeck'] = imp.fit_transform(_input1[['VRDeck']])
_input1.isnull().sum()
_input1.info()
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Num']
for i in categorical_cols:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate([_input1[i]], axis=0).astype(str)