import seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import missingno
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input1.head()
_input0.head()
print(_input1.shape)
print(_input0.shape)
_input2.head()
_input1.info()
_input0.info()
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
missingno.matrix(_input1)
missingno.matrix(_input0)
_input1.describe()
_input0.describe()
_input1[['Embarked']] = _input1[['Embarked']].fillna('Unknown')
_input1[_input1['Embarked'] == 'Unknown']
_input1[['Age']] = _input1[['Age']].fillna(99)
_input1[_input1['Age'] == 99]
_input1[['Cabin']] = _input1[['Cabin']].fillna('Unknown')
_input1[_input1['Cabin'] == 'Unknown']
_input0[['Cabin']] = _input0[['Cabin']].fillna('Unknown')
_input0[_input0['Cabin'] == 'Unknown']
_input0[['Age']] = _input0[['Age']].fillna(99)
_input0[_input0['Age'] == 99]
nan_fare = _input0[pd.isnull(_input0).any(axis=1)]
_input0[['Fare']] = _input0[['Fare']].fillna(35)
_input0[_input0['Fare'] == 35]
_input1[pd.isnull(_input1).any(axis=1)]
_input0[pd.isnull(_input0).any(axis=1)]
_input1['Cabin'] = _input1['Cabin'].str[0:2]
_input1['Cabin']
_input0['Cabin'] = _input0['Cabin'].str[0:2]
_input0['Cabin']
cabin_survived = pd.DataFrame(_input1.groupby(['Cabin', 'Pclass', 'Sex']).agg({'Survived': 'mean'}, inplace=True, index=False))
cabin_survived
cabin_survived.columns
_input1 = _input1.merge(cabin_survived, on=['Cabin', 'Pclass', 'Sex'], how='left')
_input1
_input1 = _input1.rename(columns={'Survived_x': 'Survived', 'Survived_y': 'Cabin IND'}, inplace=False)
_input1.head()
_input0.head()
_input0 = _input0.merge(cabin_survived, on=['Cabin', 'Pclass', 'Sex'], how='left')
_input0.head()
_input0 = _input0.rename(columns={'Survived': 'Cabin IND'}, inplace=False)
_input0.head()
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input0.head()
nines = _input1[_input1['Age'] == 99]
nines
classes = _input1.groupby(['Ticket']).agg({'Survived': ['mean', 'min', 'max'], 'Age': 'nunique'}, inplace=True, index=True)
classes.head()
_input1['Ticket_kind'] = _input1['Ticket'].str[0:1]
_input1['Ticket_kind']
_input0['Ticket_kind'] = _input0['Ticket'].str[0:1]
_input0['Ticket_kind']
ticket_survived = pd.DataFrame(_input1.groupby(['Ticket_kind', 'Pclass']).agg({'Ticket': 'nunique', 'Survived': 'mean'}, inplace=True, index=False))
ticket_survived
_input1 = _input1.merge(ticket_survived, on=['Ticket_kind', 'Pclass'], how='left')
_input1
_input0 = _input0.merge(ticket_survived, on=['Ticket_kind', 'Pclass'], how='left')
_input0
_input1.columns
_input1 = _input1.drop(['Ticket_x', 'Ticket_y', 'Ticket_kind'], axis=1, inplace=False)
_input1
_input1 = _input1.rename(columns={'Survived_x': 'Survived', 'Survived_y': 'Ticket IND'}, inplace=False)
_input1.head()
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input1.head()
_input0.head()
_input0 = _input0.drop(['Ticket_kind', 'Ticket_x', 'Ticket_y'], axis=1, inplace=False)
_input0['Ticket IND'] = _input0['Survived']
_input0.head()
_input0 = _input0.drop('Survived', axis=1, inplace=False)
_input0.head()
children = _input1[_input1['Age'] < 19]
children
sns.countplot(data=children, x='Pclass', hue='Survived')
sns.countplot(data=children, x='Sex', hue='Survived')
_input1['Sex'].value_counts(dropna=False)
_input1[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y='Survived', data=_input1)
plt.ylabel('Survival Probability')
plt.title('Survival Probablity by Gender')
_input1['Pclass'].value_counts(dropna=False)
_input1[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
_input0['Pclass'].value_counts(dropna=False)
sns.barplot(x='Pclass', y='Survived', data=_input1)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Passenger Class')
g = sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=_input1, kind='bar')
g.despine(left=True)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Passenger Class')
_input1['Embarked'].value_counts(dropna=False)
_input1[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Embarked', y='Survived', data=_input1)
plt.ylabel('Survival Probability')
plt.title('Survival Probability by Point of Embarkation')
_input1['Title'] = [name.split(',')[1].split('.')[0].strip() for name in _input1['Name']]
_input1[['Name', 'Title']].head()
_input1.head()
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input1.head()
_input1['Title'].value_counts()
_input1['Title'] = _input1['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Jonkheer', 'Don', 'Capt', 'the Countess', 'Sir', 'Dona'], 'Others')
_input1['Title'] = _input1['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
_input1.head()
sns.countplot(data=_input1, x='Title')
title_survived = _input1[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
title_survived
_input1 = _input1.merge(title_survived, on='Title', how='left')
_input1
_input1['Survived'] = _input1['Survived_x']
_input1['Title IND'] = _input1['Survived_y']
_input1.head()
_input1 = _input1.drop(['Survived_x', 'Title', 'Survived_y'], axis=1, inplace=False)
_input1.head()
_input0['Title'] = [name.split(',')[1].split('.')[0].strip() for name in _input0['Name']]
_input0[['Name', 'Title']].head()
_input0.head()
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input0.head()
_input0['Title'].value_counts()
_input0['Title'] = _input0['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Jonkheer', 'Don', 'Capt', 'the Countess', 'Sir', 'Dona'], 'Others')
_input0['Title'] = _input0['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
_input0.head()
_input0 = _input0.merge(title_survived, on=['Title'], how='left')
_input0
_input0 = _input0.drop('Title', axis=1, inplace=False)
_input0.head()
_input0['Title IND'] = _input0['Survived']
_input0.head()
_input0 = _input0.drop('Survived', axis=1, inplace=False)
_input0.head()
_input1.head()
_input0.loc[_input0['Age'] <= 16.0, 'Age'] = 1
_input0.loc[(_input0['Age'] > 16.0) & (_input0['Age'] <= 35.0), 'Age'] = 2
_input0.loc[(_input0['Age'] > 35.0) & (_input0['Age'] <= 50.0), 'Age'] = 3
_input0.loc[(_input0['Age'] > 50.0) & (_input0['Age'] <= 70.0), 'Age'] = 4
_input0.loc[(_input0['Age'] > 70.0) & (_input0['Age'] <= 98.0), 'Age'] = 5
_input0.loc[_input0['Age'] == 99.0, 'Age'] = 6
sns.countplot(data=_input1, x='Age')
age_sex = _input1[['Sex', 'Age', 'Survived']].groupby(['Sex', 'Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
age_sex
_input1 = _input1.merge(age_sex, on=['Sex', 'Age'], how='left')
_input1
_input0 = _input0.merge(age_sex, on=['Sex', 'Age'], how='left')
_input0
_input1['Sex Age IND'] = _input1['Survived_y']
_input1['Survived'] = _input1['Survived_x']
_input1.head()
_input1 = _input1.drop(['Survived_y', 'Sex', 'Age', 'Survived_x'], axis=1, inplace=False)
_input1.head()
sns.countplot(x='Embarked', data=_input1, hue='Survived')
sns.countplot(x='Embarked', data=_input1, hue='Pclass')
class_port = _input1[['Embarked', 'Pclass', 'Survived']].groupby(['Embarked', 'Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
class_port
_input1 = _input1.merge(class_port, on=['Embarked', 'Pclass'], how='left')
_input1
_input1['Embarked Pclass IND'] = _input1['Survived_y']
_input1['Survived'] = _input1['Survived_x']
_input1.head()
_input1 = _input1.drop(['Survived_y', 'Embarked', 'Survived_x'], axis=1, inplace=False)
_input1.head()
Sib_Parch = _input1[['SibSp', 'Parch', 'Survived']].groupby(['SibSp', 'Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Sib_Parch
_input1 = _input1.merge(Sib_Parch, on=['SibSp', 'Parch'], how='left')
_input1
_input1.columns
_input1['Family IND'] = _input1['Survived_y']
_input1.head()
_input1['Survived'] = _input1['Survived_x']
_input1.head()
_input1 = _input1.drop(['Survived_x', 'Survived_y'], axis=1, inplace=False)
_input1.head()
_input1 = _input1.drop(['SibSp', 'Parch'], axis=1, inplace=False)
_input1.head()
_input0['Sex Age IND'] = _input0['Survived']
_input0.head()
_input0 = _input0.drop(['Sex', 'Age'], inplace=False, axis=1)
_input0.head()
class_port = _input0[['Embarked', 'Pclass', 'Survived']].groupby(['Embarked', 'Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
class_port
_input0 = _input0.merge(class_port, on=['Embarked', 'Pclass'], how='left')
_input0
_input0['Embarked Pclass IND'] = _input0['Survived_y']
_input0['Survived'] = _input0['Survived_x']
_input0.head()
_input0 = _input0.drop(['Survived_y', 'Embarked', 'Survived_x'], axis=1, inplace=False)
_input0.head()
Sib_Parch_Test = _input0[['SibSp', 'Parch', 'Survived']].groupby(['SibSp', 'Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Sib_Parch_Test
_input0 = _input0.merge(Sib_Parch, on=['SibSp', 'Parch'], how='left')
_input0
_input0.columns
_input0['Family IND'] = _input0['Survived_y']
_input0.head()
_input0['Survived'] = _input0['Survived_x']
_input0.head()
_input0 = _input0.drop(['Survived_x', 'Survived_y'], axis=1, inplace=False)
_input0.head()
_input0 = _input0.drop(['SibSp', 'Parch'], axis=1, inplace=False)
_input0[_input0['Cabin IND'].isna()] = _input0['Cabin IND'].mean()
_input0[_input0['Cabin IND'].isna()]
_input0[_input0['Family IND'].isna()] = _input0['Family IND'].mean()
_input0[_input0['Family IND'].isna()]
_input0.info()
_input1.info()
_input0.head()
X_train = _input1.drop('Survived', axis=1)
Y_train = _input1['Survived']
X_test = _input0.drop('Survived', axis=1)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
X_test.head()
logreg = LogisticRegression()