import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train
df_train.columns
df_train.info()
df_train.describe()
df_train.isnull()
df_train.isnull().sum()
df_train.head()
df_train.drop('Name', axis=1, inplace=True)
df_train.dtypes
df_train.shape
df_train.ndim
df_train.size
plt.figure(figsize=(10, 6))
sns.histplot(data=df_train, x='HomePlanet', y='Destination', label=True, hue=5)
plt.figure(figsize=(15, 7))
plt.grid()
sns.histplot(data=df_train, x='Age', bins=30)
plt.figure(figsize=(10, 6))
sns.jointplot(data=df_train, x='Age', color='blue', kind='kde')
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in df_train.columns:
    if df_train[i].dtypes == 'object':
        df_train[i].fillna(df_train[i].mode()[0], inplace=True)
    else:
        df_train[i].fillna(df_train[i].median(), inplace=True)
print(df_train)
df_train['Cabin_Deck'] = df_train['Cabin'].str.split('/', expand=True)[0]
df_train['Cabin_Side'] = df_train['Cabin'].str.split('/', expand=True)[2]
df_train['Group'] = df_train['PassengerId'].str.split('_', expand=True)[0]
df_train['Num_within_Group'] = df_train['PassengerId'].str.split('_', expand=True)[1]
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df_train['Destination'].value_counts()
df_train['Cabin'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['HomePlanet'] = le.fit_transform(df_train['HomePlanet'])
df_train['CryoSleep'] = le.fit_transform(df_train['CryoSleep'])
df_train['Cabin'] = le.fit_transform(df_train['Cabin'])
df_train['VIP'] = le.fit_transform(df_train['VIP'])
df_train['Cabin_Deck'] = le.fit_transform(df_train['Cabin_Deck'])
df_train['Cabin_Side'] = le.fit_transform(df_train['Cabin_Side'])
df_train['Destination'] = le.fit_transform(df_train['Destination'])
df_train['Transported'] = le.fit_transform(df_train['Transported'])
df_train.head()
df_train.corr()
corr_matrix = df_train.corr()
(fig, ax) = plt.subplots(figsize=(20, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='RdYlBu')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.head()
df_test.columns
df_test.info()
df_test.describe()
df_test.isnull()
df_test.isnull().sum()
df_test.drop('Name', axis=1, inplace=True)
df_test.dtypes
df_test.shape
df_test.ndim
df_test.size
plt.figure(figsize=(10, 6))
sns.histplot(data=df_test, x='HomePlanet', y='Destination', label=True, hue=5)
plt.figure(figsize=(15, 7))
plt.grid()
sns.histplot(data=df_test, x='Age', bins=30)
plt.figure(figsize=(10, 6))
sns.jointplot(data=df_test, x='Age', color='blue', kind='kde')
sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in df_test.columns:
    if df_test[i].dtypes == 'object':
        df_test[i].fillna(df_test[i].mode()[0], inplace=True)
    else:
        df_test[i].fillna(df_test[i].median(), inplace=True)
print(df_test)
df_test['Cabin_Deck'] = df_test['Cabin'].str.split('/', expand=True)[0]
df_test['Cabin_Side'] = df_test['Cabin'].str.split('/', expand=True)[2]
df_test['Group'] = df_test['PassengerId'].str.split('_', expand=True)[0]
df_test['Num_within_Group'] = df_test['PassengerId'].str.split('_', expand=True)[1]
sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df_test['Destination'].value_counts()
df_test['Cabin'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_test['HomePlanet'] = le.fit_transform(df_test['HomePlanet'])
df_test['CryoSleep'] = le.fit_transform(df_test['CryoSleep'])
df_test['Cabin'] = le.fit_transform(df_test['Cabin'])
df_test['VIP'] = le.fit_transform(df_test['VIP'])
df_test['Cabin_Deck'] = le.fit_transform(df_test['Cabin_Deck'])
df_test['Cabin_Side'] = le.fit_transform(df_test['Cabin_Side'])
df_test['Destination'] = le.fit_transform(df_test['Destination'])
df_test.head()
df_test.corr()
corr_matrix = df_test.corr()
(fig, ax) = plt.subplots(figsize=(20, 10))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='RdYlBu')
from sklearn.model_selection import train_test_split
df_train.columns
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
rfr = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2)