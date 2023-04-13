import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.describe()
import matplotlib.pyplot as plt
import seaborn as sns
corr = _input1.corr()
sns.heatmap(corr, annot=True)
_input1['FoodCourt_cat'] = pd.qcut(_input1['FoodCourt'], 20, duplicates='drop')
_input0['FoodCourt_cat'] = pd.qcut(_input0['FoodCourt'], 20, duplicates='drop')
_input1['Spa_cat'] = pd.qcut(_input1['Spa'], 20, duplicates='drop')
_input0['Spa_cat'] = pd.qcut(_input0['Spa'], 20, duplicates='drop')
_input1['VRDeck_cat'] = pd.qcut(_input1['VRDeck'], 20, duplicates='drop')
_input0['VRDeck_cat'] = pd.qcut(_input0['VRDeck'], 20, duplicates='drop')
cat_cols = ['FoodCourt_cat', 'Spa_cat', 'VRDeck_cat']
num_cols = ['FoodCourt', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(40, 10))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=_input1, ax=axs[idx1][idx2])
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(cat_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.barplot(x=row, y=col, data=_input1, ax=axs[idx1][idx2])
print(_input1.isnull().sum())
print(_input0.isnull().sum())
_input1['Expenditure'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['Expenditure'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1.loc[_input1['RoomService'] > 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0.loc[_input0['RoomService'] > 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1.loc[_input1['FoodCourt'] > 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0.loc[_input0['FoodCourt'] > 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1.loc[_input1['ShoppingMall'] > 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0.loc[_input0['ShoppingMall'] > 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1.loc[_input1['Spa'] > 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0.loc[_input0['Spa'] > 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1.loc[_input1['VRDeck'] > 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0.loc[_input0['VRDeck'] > 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1.loc[_input1['Expenditure'] == 0, 'CryoSleep'] = _input1['CryoSleep'].fillna(True)
_input0.loc[_input0['Expenditure'] == 0, 'CryoSleep'] = _input0['CryoSleep'].fillna(True)
print(_input1.isnull().sum())
print(_input0.isnull().sum())
_input1[_input1['CryoSleep'].isnull()]
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(True)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(True)
print(_input1.isnull().sum())
print(_input0.isnull().sum())
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.scatterplot(x='Transported', y=col, data=_input1, ax=axs[idx])
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 3))
for (idx, col) in enumerate(num_cols):
    sns.histplot(_input1[col], bins=100, ax=axs[idx])
outlier_r = _input1[_input1['RoomService'] > 8000].index
_input1 = _input1.drop(outlier_r, axis=0)
outlier_f = _input1[_input1['FoodCourt'] > 20000].index
_input1 = _input1.drop(outlier_f, axis=0)
outlier_sm = _input1[_input1['ShoppingMall'] > 9000].index
_input1 = _input1.drop(outlier_sm, axis=0)
outlier_s = _input1[_input1['Spa'] > 12000].index
_input1 = _input1.drop(outlier_s, axis=0)
outlier_v = _input1[_input1['VRDeck'] > 14000].index
_input1 = _input1.drop(outlier_v, axis=0)
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.scatterplot(x='Transported', y=col, data=_input1, ax=axs[idx])
(fig, axs) = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(30, 3))
for (idx, col) in enumerate(num_cols):
    sns.histplot(_input1[col], bins=100, ax=axs[idx])
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(num_cols), ncols=len(num_cols), figsize=(30, 20))
for (idx1, row) in enumerate(num_cols):
    for (idx2, col) in enumerate(num_cols):
        sns.scatterplot(x=row, y=col, data=_input1, ax=axs[idx1][idx2])

def age_category(age):
    cat = ''
    if age <= 12:
        cat = '0~12'
    elif age <= 17:
        cat = '13~17'
    elif age <= 25:
        cat = '17~25'
    elif age <= 30:
        cat = '26~30'
    elif age <= 50:
        cat = '31~50'
    else:
        cat = '51~'
    return cat
_input1['Age_cat'] = _input1['Age'].apply(lambda x: age_category(x))
_input0['Age_cat'] = _input0['Age'].apply(lambda x: age_category(x))
cat_cols = ['Age_cat']
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(23, 3))
for (idx, col) in enumerate(num_cols):
    sns.barplot(x='Age_cat', y=col, data=_input1, ax=axs[idx])
_input1.isnull().sum()
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    _input1.loc[_input1['CryoSleep'] == True, col] = _input1[col].fillna(0)
    _input0.loc[_input0['CryoSleep'] == True, col] = _input0[col].fillna(0)
for col in num_cols:
    _input1.loc[_input1['Age'] <= 12, col] = _input1[col].fillna(0)
    _input0.loc[_input0['Age'] <= 12, col] = _input0[col].fillna(0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1.groupby('VRDeck_cat')['FoodCourt'].transform('mean'))
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0.groupby('VRDeck_cat')['FoodCourt'].transform('mean'))
_input1['Spa'] = _input1['Spa'].fillna(_input1.groupby('Transported')['Spa'].transform('mean'))
_input0['Spa'] = _input0['Spa'].fillna(_input0.groupby('FoodCourt_cat')['Spa'].transform('mean'))
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1.groupby('Transported')['VRDeck'].transform('mean'))
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0.groupby('FoodCourt_cat')['VRDeck'].transform('mean'))
_input1['RoomService'] = _input1['RoomService'].fillna(_input1.groupby('Transported')['RoomService'].transform('mean'))
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].mean())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean())
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].mean())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean())
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].mean())
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean())
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].mean())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean())
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].mean())
_input1['Group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0])
_input0['Group'] = _input0['PassengerId'].apply(lambda x: x.split('_')[0])
import matplotlib.pyplot as plt
import seaborn as sns
GHP_gb = _input1.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
sns.countplot((GHP_gb > 0).sum(axis=1))

def cabin_cat(c):
    if c[0] == 'A':
        c = 'A'
    elif c[0] == 'B':
        c = 'B'
    elif c[0] == 'C':
        c = 'C'
    elif c[0] == 'D':
        c = 'D'
    elif c[0] == 'E':
        c = 'E'
    elif c[0] == 'F':
        c = 'F'
    elif c[0] == 'G':
        c = 'G'
    else:
        c = 'T'
    return c
cabin_train = _input1[_input1['Cabin'].notnull()]
cabin_test = _input0[_input0['Cabin'].notnull()]
_input1['Cabin_cat'] = cabin_train['Cabin'].apply(lambda x: cabin_cat(x))
_input0['Cabin_cat'] = cabin_test['Cabin'].apply(lambda x: cabin_cat(x))
_input1.groupby('Cabin_cat')[num_cols].mean()
(fig, axs) = plt.subplots(nrows=len(cat_cols), ncols=len(num_cols), figsize=(30, 4))
for (idx, col) in enumerate(num_cols):
    sns.barplot(x='Cabin_cat', y=col, data=_input1, ax=axs[idx])
sns.barplot(x='Cabin_cat', y='Transported', data=_input1)
pd.crosstab(index=_input1['HomePlanet'], columns=_input1['Cabin_cat'])
_input1.loc[_input1['HomePlanet'].isnull() & _input1['Cabin_cat'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
_input1.loc[_input1['HomePlanet'].isnull() & (_input1['Cabin_cat'] == 'G'), 'HomePlanet'] = 'Earth'
_input0.loc[_input0['HomePlanet'].isnull() & _input0['Cabin_cat'].isin(['A', 'B', 'C', 'T']), 'HomePlanet'] = 'Europa'
_input0.loc[_input0['HomePlanet'].isnull() & (_input0['Cabin_cat'] == 'G'), 'HomePlanet'] = 'Earth'
_input1.isnull().sum()
_input1.loc[_input1['Cabin_cat'] != 'D', 'HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input0.loc[_input0['Cabin_cat'] != 'D', 'HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input1.isnull().sum()
_input1.loc[_input1['Cabin_cat'] == 'D', 'HomePlanet'] = _input1['HomePlanet'].fillna('Mars')
_input0.loc[_input0['Cabin_cat'] == 'D', 'HomePlanet'] = _input0['HomePlanet'].fillna('Mars')
_input1.isnull().sum()
_input1['Expenditure'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['Expenditure'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1.loc[(_input1['CryoSleep'] == False) & (_input1['Expenditure'] == 0), 'Age'] = _input1['Age'].fillna(5)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input0.loc[(_input0['CryoSleep'] == False) & (_input0['Expenditure'] == 0), 'Age'] = _input0['Age'].fillna(5)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0])
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0])
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0])
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0])
_input1 = _input1.drop(['PassengerId', 'Name', 'Cabin', 'Cabin_cat', 'Age_cat', 'FoodCourt_cat', 'Spa_cat', 'VRDeck_cat', 'Expenditure', 'Group'], axis=1)
_input0 = _input0.drop(['PassengerId', 'Name', 'Cabin', 'Cabin_cat', 'Age_cat', 'FoodCourt_cat', 'Spa_cat', 'VRDeck_cat', 'Expenditure', 'Group'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
_input1['VIP'] = le.fit_transform(_input1['VIP'])
_input0['CryoSleep'] = le.fit_transform(_input0['CryoSleep'])
_input0['VIP'] = le.fit_transform(_input0['VIP'])
cols = ['HomePlanet', 'Destination']
df_oh = pd.get_dummies(_input1[cols], drop_first=True)
_input1 = pd.concat([_input1, df_oh], axis=1)
_input1 = _input1.drop(cols, axis=1)
test_oh = pd.get_dummies(_input0[cols], drop_first=True)
_input0 = pd.concat([_input0, test_oh], axis=1)
_input0 = _input0.drop(cols, axis=1)
df_y = _input1['Transported']
_input1 = _input1.drop('Transported', axis=1)
_input1 = pd.concat([_input1, df_y], axis=1)
y_df = _input1['Transported']
x_df = _input1.drop('Transported', axis=1)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_df, y_df, test_size=0.2, random_state=156)
(x_tr, x_val, y_tr, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=156)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
dt_clf = DecisionTreeClassifier(random_state=11, max_depth=11, min_samples_split=120)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(solver='liblinear')