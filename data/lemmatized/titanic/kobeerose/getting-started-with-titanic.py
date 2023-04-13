import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
print(_input1.shape)
print(_input0.shape)
_input1.describe()
_input1.describe()
_input1.info()
_input0.info()
print('training data\n', _input1.isnull().sum())
print('\ntesting data\n', _input0.isnull().sum())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
count_plt = sns.countplot(_input1['Survived'])
Sex_plt = sns.countplot(x='Survived', data=_input1, hue='Sex')
Embarked_plt = sns.countplot(x='Survived', data=_input1, hue='Embarked')
Pclass_plt = sns.countplot(x='Survived', data=_input1, hue='Pclass')
SibSp_plt = sns.boxplot(x='SibSp', y='Survived', data=_input1)
Parch_plt = sns.boxplot(x='Parch', y='Survived', data=_input1)
Age_plt = sns.distplot(_input1['Age'])
(f, ax) = plt.subplots(figsize=(10, 8))
corr = _input1.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool_), linewidths=0.1, annot=True, cmap=sns.diverging_palette(150, 10, as_cmap=True), square=True, ax=ax)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(_input1['Fare'].mean(), inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.drop(columns=['Cabin', 'Name', 'Ticket'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['Cabin', 'Name', 'Ticket'], axis=1, inplace=False)
_input1 = _input1.drop(['PassengerId'], axis=1)
_input1.loc[_input1.Sex == 'female', 'Sex'] = 1
_input1.loc[_input1.Sex == 'male', 'Sex'] = 0
_input1['Sex'] = _input1['Sex'].astype(str).astype(float)
_input1.loc[_input1.Embarked == 'S', 'Embarked'] = 3
_input1.loc[_input1.Embarked == 'C', 'Embarked'] = 2
_input1.loc[_input1.Embarked == 'Q', 'Embarked'] = 1
_input1['Embarked'] = _input1['Embarked'].astype(str).astype(float)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].mean())
_input0.loc[_input0.Sex == 'female', 'Sex'] = 1
_input0.loc[_input0.Sex == 'male', 'Sex'] = 0
_input0['Sex'] = _input0['Sex'].astype(str).astype(float)
_input0.loc[_input0.Embarked == 'S', 'Embarked'] = 3
_input0.loc[_input0.Embarked == 'C', 'Embarked'] = 2
_input0.loc[_input0.Embarked == 'Q', 'Embarked'] = 1
_input0['Embarked'] = _input0['Embarked'].astype(str).astype(float)
_input0.isnull().sum()
print(_input1.head())
print(_input0.head())
print(_input1.corr())
_input1.info()
_input0.info()
train_x = _input1.drop(columns=['Survived'], axis=1)
train_y = _input1['Survived']
test_x = _input0.drop('PassengerId', axis=1)
logistic = LogisticRegression(solver='liblinear')