import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.shape
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
train.describe()
train.HomePlanet.value_counts()
train.Destination.value_counts()
sns.catplot(x='HomePlanet', y='Transported', hue='Destination', kind='bar', data=train, height=5, aspect=2)
sns.catplot(x='Destination', y='Transported', hue='HomePlanet', kind='bar', data=train, height=5, aspect=2)
sns.catplot(x='Destination', y='Transported', kind='bar', data=train, height=5, aspect=2)
sns.heatmap(train.corr(), annot=True)
train.CryoSleep.value_counts()
cryo_transport = train[['CryoSleep', 'Transported']]

cryo_transport = pd.DataFrame(train.groupby('CryoSleep')['Transported'].sum())
cryo_transport
vip_transport = pd.DataFrame(train.groupby('VIP')['Transported'].sum())
vip_transport
train[['PassengerNo', 'GroupMemNo']] = train['PassengerId'].str.split(pat='_', expand=True)
train.head()
train['PartySize'] = train.groupby('PassengerNo')['PassengerNo'].transform('count')
train.head()
train.PartySize.value_counts()
train['PartySize'] = train['PartySize'].astype('category')
(fig, ax) = plt.subplots(figsize=(12, 7))
ax = sns.countplot(x='PartySize', hue='Transported', data=train)
test[['PassengerNo', 'GroupMemNo']] = test['PassengerId'].str.split(pat='_', expand=True)
test['PartySize'] = test.groupby('PassengerNo')['PassengerNo'].transform('count')
test['PartySize'] = test['PartySize'].astype('category')
train.Transported.value_counts()
party_size = pd.DataFrame(train.groupby('PartySize')['Transported'].mean().reset_index())
party_size
train[['cabin1', 'cabin2', 'cabin3']] = train['Cabin'].str.split(pat='/', expand=True)
test[['cabin1', 'cabin2', 'cabin3']] = test['Cabin'].str.split(pat='/', expand=True)
train.head()
sns.catplot(x='cabin1', y='Transported', kind='bar', data=train, height=5, aspect=3)
sns.catplot(x='cabin3', y='Transported', kind='bar', data=train, height=5, aspect=3)
train.cabin2.value_counts()
train.columns
passenger_id = test[['PassengerId']]
passenger_id.head()
X = train.drop(['Transported', 'PassengerId', 'Cabin', 'Name', 'PassengerNo', 'GroupMemNo', 'cabin2'], axis=1)
y = train['Transported']
X_test = test.drop(['PassengerId', 'Cabin', 'Name', 'PassengerNo', 'GroupMemNo', 'cabin2'], axis=1)
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