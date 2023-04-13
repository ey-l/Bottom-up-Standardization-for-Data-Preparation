import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.columns
_input1.info()
_input1.describe()
_input1.isnull()
_input1.isnull().sum()
_input1.head()
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input1.dtypes
_input1.shape
_input1.ndim
_input1.size
plt.figure(figsize=(10, 6))
sns.histplot(data=_input1, x='HomePlanet', y='Destination', label=True, hue=5)
plt.figure(figsize=(15, 7))
plt.grid()
sns.histplot(data=_input1, x='Age', bins=30)
plt.figure(figsize=(10, 6))
sns.jointplot(data=_input1, x='Age', color='blue', kind='kde')
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in _input1.columns:
    if _input1[i].dtypes == 'object':
        _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
    else:
        _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
print(_input1)
_input1['Cabin_Deck'] = _input1['Cabin'].str.split('/', expand=True)[0]
_input1['Cabin_Side'] = _input1['Cabin'].str.split('/', expand=True)[2]
_input1['Group'] = _input1['PassengerId'].str.split('_', expand=True)[0]
_input1['Num_within_Group'] = _input1['PassengerId'].str.split('_', expand=True)[1]
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
_input1['Destination'].value_counts()
_input1['Cabin'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['HomePlanet'] = le.fit_transform(_input1['HomePlanet'])
_input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
_input1['Cabin'] = le.fit_transform(_input1['Cabin'])
_input1['VIP'] = le.fit_transform(_input1['VIP'])
_input1['Cabin_Deck'] = le.fit_transform(_input1['Cabin_Deck'])
_input1['Cabin_Side'] = le.fit_transform(_input1['Cabin_Side'])
_input1['Destination'] = le.fit_transform(_input1['Destination'])
_input1['Transported'] = le.fit_transform(_input1['Transported'])
_input1.head()
_input1.corr()
corr_matrix = _input1.corr()
(fig, ax) = plt.subplots(figsize=(20, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='RdYlBu')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0.columns
_input0.info()
_input0.describe()
_input0.isnull()
_input0.isnull().sum()
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input0.dtypes
_input0.shape
_input0.ndim
_input0.size
plt.figure(figsize=(10, 6))
sns.histplot(data=_input0, x='HomePlanet', y='Destination', label=True, hue=5)
plt.figure(figsize=(15, 7))
plt.grid()
sns.histplot(data=_input0, x='Age', bins=30)
plt.figure(figsize=(10, 6))
sns.jointplot(data=_input0, x='Age', color='blue', kind='kde')
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in _input0.columns:
    if _input0[i].dtypes == 'object':
        _input0[i] = _input0[i].fillna(_input0[i].mode()[0], inplace=False)
    else:
        _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
print(_input0)
_input0['Cabin_Deck'] = _input0['Cabin'].str.split('/', expand=True)[0]
_input0['Cabin_Side'] = _input0['Cabin'].str.split('/', expand=True)[2]
_input0['Group'] = _input0['PassengerId'].str.split('_', expand=True)[0]
_input0['Num_within_Group'] = _input0['PassengerId'].str.split('_', expand=True)[1]
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False, cmap='viridis')
_input0['Destination'].value_counts()
_input0['Cabin'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input0['HomePlanet'] = le.fit_transform(_input0['HomePlanet'])
_input0['CryoSleep'] = le.fit_transform(_input0['CryoSleep'])
_input0['Cabin'] = le.fit_transform(_input0['Cabin'])
_input0['VIP'] = le.fit_transform(_input0['VIP'])
_input0['Cabin_Deck'] = le.fit_transform(_input0['Cabin_Deck'])
_input0['Cabin_Side'] = le.fit_transform(_input0['Cabin_Side'])
_input0['Destination'] = le.fit_transform(_input0['Destination'])
_input0.head()
_input0.corr()
corr_matrix = _input0.corr()
(fig, ax) = plt.subplots(figsize=(20, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='RdYlBu')
from sklearn.model_selection import train_test_split
_input1.columns
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
rfr = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2)