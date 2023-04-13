import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1 = _input1.drop(['PassengerId', 'Ticket'], axis=1, inplace=False)
_input1.describe()
_input1.isnull().sum()
import re
_input1['Salutations'] = _input1['Name'].str.extract('([A-Z]{1}[a-z]+\\.)')
_input1['Salutations'].unique()
_input1[_input1['Salutations'] == 'Master.']['Age'].median()
_input1[(_input1['Salutations'] == 'Miss.') | (_input1['Salutations'] == 'Ms.') | (_input1['Salutations'] == 'Mlle.')]['Age'].median()
_input1[_input1['Salutations'] == 'Mr.']['Age'].median()
_input1[(_input1['Salutations'] == 'Mrs.') | (_input1['Salutations'] == 'Mme.')]['Age'].median()
master = _input1['Salutations'] == 'Master.'
miss = (_input1['Salutations'] == 'Miss.') | (_input1['Salutations'] == 'Ms.') | (_input1['Salutations'] == 'Mlle.')
mister = _input1['Salutations'] == 'Mr.'
missus = (_input1['Salutations'] == 'Mrs.') | (_input1['Salutations'] == 'Mme.')
_input1['Title'] = 'Others'
_input1['Title'][master] = 'Master'
_input1['Title'][miss] = 'Miss'
_input1['Title'][mister] = 'Mister'
_input1['Title'][missus] = 'Missus'
_input1['Age'] = _input1.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))
_input1['Cabin'] = _input1['Cabin'].str.extract('([A-Z]{1})').fillna('Z')
_input1['Embarked'].mode()
_input1['Embarked'] = _input1['Embarked'].fillna('S')
_input1.info()
_input1 = _input1.drop(['Name', 'Salutations'], axis=1, inplace=False)
_input1.head()
plt.figure(figsize=(8, 8))
sns.countplot('Pclass', data=_input1, hue='Survived')
(fig, axes) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
ax = sns.countplot('Sex', data=_input1, hue='Survived', ax=axes[0])
ax1 = _input1['Sex'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
sex = {'male': 0, 'female': 1}
_input1['Sex'] = _input1['Sex'].map(sex)
plt.figure(figsize=(8, 8))
survived = _input1[_input1['Survived'] == 1]
not_survived = _input1[_input1['Survived'] == 0]
sns.distplot(survived['Age'], kde=False, label='Survived')
sns.distplot(not_survived['Age'], kde=False, label='Did not survive')
plt.legend()
(fig, axes) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
men = _input1[_input1['Sex'] == 0]
women = _input1[_input1['Sex'] == 1]
ax = sns.distplot(women[women['Survived'] == 1]['Age'], label='Survived', ax=axes[0], kde=False, bins=20)
ax1 = sns.distplot(women[women['Survived'] == 0]['Age'], label='Did not survive', ax=axes[0], kde=False, bins=20)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived'] == 1]['Age'], label='Survived', ax=axes[1], kde=False, bins=20)
ax1 = sns.distplot(men[men['Survived'] == 0]['Age'], label='Did not survive', ax=axes[1], kde=False, bins=20)
ax.legend()
ax.set_title('Male')
pd.qcut(_input1[_input1['Age'] > 8]['Age'], 5)
age = _input1['Age'] < 8
age1 = (_input1['Age'] >= 9) & (_input1['Age'] < 21)
age2 = (_input1['Age'] >= 21) & (_input1['Age'] < 28)
age3 = (_input1['Age'] >= 28) & (_input1['Age'] < 30)
age4 = (_input1['Age'] >= 28) & (_input1['Age'] < 39)
age5 = _input1['Age'] >= 39
_input1['Age'][age] = 0
_input1['Age'][age1] = 1
_input1['Age'][age2] = 2
_input1['Age'][age3] = 3
_input1['Age'][age4] = 4
_input1['Age'][age5] = 5
plt.figure(figsize=(10, 10))
_input1['Family'] = _input1['SibSp'] + _input1['Parch']
survived = _input1[_input1['Survived'] == 1]
not_survived = _input1[_input1['Survived'] == 0]
sns.distplot(survived['Family'], kde=False, label='Survived', bins=50)
sns.distplot(not_survived['Family'], kde=False, label='Did not survive', bins=50)
plt.legend()
none = _input1['Family'] == 0
four = _input1['Family'] >= 4
_input1['Fam_Cat'] = 1
_input1['Fam_Cat'][none] = 0
_input1['Fam_Cat'][four] = 2
plt.figure(figsize=(10, 10))
survived = _input1[_input1['Survived'] == 1]
not_survived = _input1[_input1['Survived'] == 0]
sns.distplot(survived['Fare'], kde=False, label='Survived', bins=100, color='green')
sns.distplot(not_survived['Fare'], kde=False, label='Did not survive', bins=100, color='red')
plt.legend()
pd.qcut(_input1['Fare'], 6)
fare = _input1['Fare'] < 7.775
fare1 = (_input1['Fare'] >= 7.775) & (_input1['Fare'] < 8.662)
fare2 = (_input1['Fare'] >= 8.662) & (_input1['Fare'] < 14.454)
fare3 = (_input1['Fare'] >= 14.454) & (_input1['Fare'] < 26)
fare4 = (_input1['Fare'] >= 26) & (_input1['Fare'] < 52.369)
fare5 = _input1['Fare'] >= 52.369
_input1['Fare'][fare] = 0
_input1['Fare'][fare1] = 1
_input1['Fare'][fare2] = 2
_input1['Fare'][fare3] = 3
_input1['Fare'][fare4] = 4
_input1['Fare'][fare5] = 5
sns.catplot(x='Sex', y='Survived', kind='bar', data=_input1, hue='Embarked', palette='rocket', aspect=1.3)
port = {'S': 1, 'C': 2, 'Q': 3}
_input1['Embarked'] = _input1['Embarked'].map(port)
_input1['Cabin'].unique()
cabin = {'Z': 8, 'T': 7, 'G': 6, 'F': 5, 'E': 4, 'D': 3, 'C': 2, 'B': 1, 'A': 0}
_input1['Cabin'] = _input1['Cabin'].map(cabin)
_input1.head()
df = _input1.drop(['SibSp', 'Parch', 'Family'], axis=1)
df.head()
title = {'Master': 0, 'Miss': 1, 'Mister': 3, 'Missus': 4, 'Others': 5}
df['Title'] = df['Title'].map(title)
df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(int)
df['Pclass_Sex'] = df['Pclass'] * df['Sex']
df['Pclass_Age'] = df['Pclass'] * df['Age']
df['Pclass_Fare'] = df['Pclass'] * df['Fare']
df['Sex_Age'] = df['Sex'] * df['Age']
df['Sex_Fare'] = df['Sex'] * df['Fare']
df['Age_Fare'] = df['Age'] * df['Fare']
df.head()
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2)
from xgboost import XGBClassifier
classifier = XGBClassifier(nthread=1, colsample_bytree=0.8, learning_rate=0.03, max_depth=4, min_child_weight=2, n_estimators=1000, subsample=0.8)