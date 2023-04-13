import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split, GridSearchCV
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
train = _input1.copy()
test = _input0.copy()
train.head()
test.head()
train.info()
train.shape
train.dtypes
train.describe().T
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass', y='Survived', data=train)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='SibSp', y='Survived', data=train)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Parch', y='Survived', data=train)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y='Survived', data=train)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
train['Ticket']
train.head()
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train.head()
train.describe().T
sns.boxplot(x=train['Fare'])
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
lower_limit
upper_limit = Q3 + 1.5 * IQR
upper_limit
train['Fare'] > upper_limit
train.sort_values('Fare', ascending=False).head()
train['Fare'] = train['Fare'].replace(512.3292, 300)
train.sort_values('Fare', ascending=False).head()
test.sort_values('Fare', ascending=False)
test['Fare'] = test['Fare'].replace(512.3292, 300)
test.sort_values('Fare', ascending=False)
train.isnull().sum()
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
train.isnull().sum()
test.isnull().sum()
train.isnull().sum()
test.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
train.isnull().sum()
test.isnull().sum()
test[test['Fare'].isnull()]
test[['Pclass', 'Fare']].groupby('Pclass').mean()
test['Fare'] = test['Fare'].fillna(12)
test['Fare'].isnull().sum()
train.head()
train['CabinBool'] = train['Cabin'].notnull().astype('int')
test['CabinBool'] = test['Cabin'].notnull().astype('int')
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train.head()
train.isnull().sum()
test.isnull().sum()
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
train.head()
from sklearn import preprocessing
lbe = preprocessing.LabelEncoder()
train['Sex'] = lbe.fit_transform(train['Sex'])
test['Sex'] = lbe.fit_transform(test['Sex'])
train.head()
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
train.head()
train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')
train.head()
test.head()
train[['Title', 'PassengerId']].groupby('Title').count()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 5}
train['Title'] = train['Title'].map(title_mapping)
train.isnull().sum()
test['Title'] = test['Title'].map(title_mapping)
test.head()
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train.head()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=mylabels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=mylabels)
train.head()
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)
train.head()
train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
train.head()
train.head()
train['FamilySize'] = _input1['SibSp'] + _input1['Parch'] + 1
test['FamilySize'] = _input0['SibSp'] + _input0['Parch'] + 1
train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if s == 2 else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if s == 2 else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
test.head()
train = pd.get_dummies(train, columns=['Title'])
train = pd.get_dummies(train, columns=['Embarked'], prefix='Em')
train.head()
test = pd.get_dummies(test, columns=['Title'])
test = pd.get_dummies(test, columns=['Embarked'], prefix='Em')
test.head()
train['Pclass'] = train['Pclass'].astype('category')
train = pd.get_dummies(train, columns=['Pclass'], prefix='Pc')
train.head()
test['Pclass'] = test['Pclass'].astype('category')
test = pd.get_dummies(test, columns=['Pclass'], prefix='Pc')
train.head()
test.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train['Survived']
(x_train, x_test, y_train, y_test) = train_test_split(predictors, target, test_size=0.2, random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression
import pandas as pd
logreg = LogisticRegression()