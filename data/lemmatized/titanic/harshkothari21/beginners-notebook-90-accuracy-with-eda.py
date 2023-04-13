import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from statistics import mode
import re
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1.isnull().sum()
sns.countplot(_input1['Survived'])
sns.countplot(x='Survived', hue='Sex', data=_input1)
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='plasma')
_input1.describe()
sns.countplot(_input1['Pclass'])
_input1.Name.value_counts().head()
_input1['Age'].hist(bins=40)
_input1['SibSp'].value_counts()
sns.countplot(_input1['SibSp'])
plt.title('Count plot for SibSp')
sns.countplot(_input1['Parch'])
plt.title('Count plot for Parch')
_input1.Ticket.value_counts(dropna=False, sort=True).head()
_input1['Fare'].hist(bins=50)
plt.ylabel('Price')
plt.xlabel('Index')
plt.title('Fare Price distribution')
_input1.Cabin.value_counts(0)
sns.countplot(_input1['Embarked'])
plt.title('Count plot for Embarked')
sns.heatmap(_input1.corr(), annot=True)
sns.countplot(x='Survived', hue='Pclass', data=_input1)
plt.title('Count plot for Pclass categorized by Survived')
age_group = _input1.groupby('Pclass')['Age']
age_group.median()
age_group.mean()
_input1.loc[_input1.Age.isnull(), 'Age'] = _input1.groupby('Pclass').Age.transform('median')
_input1['Age'].isnull().sum()
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='plasma')
_input1['Sex'][_input1['Sex'] == 'male'] = 0
_input1['Sex'][_input1['Sex'] == 'female'] = 1
_input1['Embarked'][_input1['Embarked'] == 'S'] = 0
_input1['Embarked'][_input1['Embarked'] == 'C'] = 1
_input1['Embarked'][_input1['Embarked'] == 'Q'] = 2
_input1.head()
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0['Survived'] = np.nan
full = pd.concat([_input1, _input0])
full.isnull().sum()
full.head()
full['Embarked'] = full['Embarked'].fillna(mode(full['Embarked']))
full['Sex'][full['Sex'] == 'male'] = 0
full['Sex'][full['Sex'] == 'female'] = 1
full['Embarked'][full['Embarked'] == 'S'] = 0
full['Embarked'][full['Embarked'] == 'C'] = 1
full['Embarked'][full['Embarked'] == 'Q'] = 2
sns.heatmap(full.corr(), annot=True)
full['Age'] = full.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
full.isnull().sum()
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
full['Cabin'] = full['Cabin'].fillna('U')
full['Cabin'].unique().tolist()[:20]
full['Cabin'] = full['Cabin'].map(lambda x: re.compile('([a-zA-Z])').search(x).group())
full['Cabin'].unique().tolist()
cabin_category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 9}
full['Cabin'] = full['Cabin'].map(cabin_category)
full['Cabin'].unique().tolist()
full['Name'].head()
full['Name'] = full.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
full['Name'].unique().tolist()
full['Name'].value_counts(normalize=True) * 100
full = full.rename(columns={'Name': 'Title'}, inplace=False)
full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
full['Title'].value_counts(normalize=True) * 100
title_category = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}
full['Title'] = full['Title'].map(title_category)
full['Title'].unique().tolist()
full['familySize'] = full['SibSp'] + full['Parch'] + 1
full = full.drop(['SibSp', 'Parch', 'Ticket'], axis=1)
full.head()
_input0 = full[full['Survived'].isna()].drop(['Survived'], axis=1)
_input0.head()
_input1 = full[full['Survived'].notna()]
_input1['Survived'] = _input1['Survived'].astype(np.int8)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop(['Survived', 'PassengerId'], axis=1), _input1['Survived'], test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression
LogisticRegression = LogisticRegression(max_iter=10000)