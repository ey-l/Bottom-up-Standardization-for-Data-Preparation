import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.shape
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
_input1.describe()
_input1.HomePlanet.value_counts()
_input1.Destination.value_counts()
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', data=_input1, height=5, aspect=2)
sns.catplot(x='Destination', y='Transported', hue='HomePlanet', kind='bar', data=_input1, height=5, aspect=2)
sns.catplot(x='Destination', y='Transported', kind='bar', data=_input1, height=5, aspect=2)
sns.heatmap(_input1.corr(), annot=True)
_input1.CryoSleep.value_counts()
cryo_transport = _input1[['CryoSleep', 'Transported']]
cryo_transport = pd.DataFrame(_input1.groupby('CryoSleep')['Transported'].sum())
cryo_transport
vip_transport = pd.DataFrame(_input1.groupby('VIP')['Transported'].sum())
vip_transport
_input1[['PassengerNo', 'GroupMemNo']] = _input1['PassengerId'].str.split(pat='_', expand=True)
_input1.head()
_input1['PartySize'] = _input1.groupby('PassengerNo')['PassengerNo'].transform('count')
_input1.head()
_input1.PartySize.value_counts()
_input1['PartySize'] = _input1['PartySize'].astype('category')
(fig, ax) = plt.subplots(figsize=(12, 7))
ax = sns.countplot(x='PartySize', hue='Transported', data=_input1)
_input0[['PassengerNo', 'GroupMemNo']] = _input0['PassengerId'].str.split(pat='_', expand=True)
_input0['PartySize'] = _input0.groupby('PassengerNo')['PassengerNo'].transform('count')
_input0['PartySize'] = _input0['PartySize'].astype('category')
_input1.Transported.value_counts()
party_size = pd.DataFrame(_input1.groupby('PartySize')['Transported'].mean().reset_index())
party_size
_input1[['cabin1', 'cabin2', 'cabin3']] = _input1['Cabin'].str.split(pat='/', expand=True)
_input0[['cabin1', 'cabin2', 'cabin3']] = _input0['Cabin'].str.split(pat='/', expand=True)
_input1.head()
sns.catplot(x='cabin1', y='Transported', kind='bar', data=_input1, height=5, aspect=3)
sns.catplot(x='cabin3', y='Transported', kind='bar', data=_input1, height=5, aspect=3)
_input1.cabin2.value_counts()
_input1.columns
passenger_id = _input0[['PassengerId']]
passenger_id.head()
X = _input1.drop(['Transported', 'PassengerId', 'Cabin', 'Name', 'PassengerNo', 'GroupMemNo', 'cabin2'], axis=1)
y = _input1['Transported']
X_test = _input0.drop(['PassengerId', 'Cabin', 'Name', 'PassengerNo', 'GroupMemNo', 'cabin2'], axis=1)
print(X.shape, X_test.shape)
mode_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin1', 'cabin3', 'PartySize']
X[mode_cols].isna().sum()
X[mode_cols] = X[mode_cols].fillna(X.mode().iloc[0])
X[mode_cols].isna().sum()
float_cols = [col for col in X.columns if X[col].dtype == 'float64']
X[float_cols].isna().sum()
X[float_cols].mean()
X[float_cols] = X[float_cols].fillna(X.mean().iloc[0])
X[float_cols].isna().sum()
X_test.isna().sum()
X_test[mode_cols] = X_test[mode_cols].fillna(X_test.mode().iloc[0])
X_test[float_cols] = X_test[float_cols].fillna(X_test.mean().iloc[0])
X_test.isna().any()
X_oh = pd.get_dummies(X, columns=mode_cols)
X_test_oh = pd.get_dummies(X_test, columns=mode_cols)
X_oh.head()
X_test_oh.shape
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
scaled_features = mmsc.fit_transform(X_oh)
X_oh_scl = pd.DataFrame(scaled_features, index=X_oh.index, columns=X_oh.columns)
X_oh_scl.head()
scaled_features = mmsc.transform(X_test_oh)
X_test_oh_scl = pd.DataFrame(scaled_features, index=X_test_oh.index, columns=X_test_oh.columns)
X_test_oh_scl.head()
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
(X_train, X_valid, y_train, y_valid) = train_test_split(X_oh_scl, y, test_size=0.3, random_state=43)
print(X_train.shape, X_valid.shape, len(y_train), len(y_valid))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
svc = SVC()
grid = GridSearchCV(svc, param_grid=param_grid, refit=True, verbose=2)
svc = SVC(C=100, gamma=0.1, kernel='rbf')
from sklearn.metrics import accuracy_score