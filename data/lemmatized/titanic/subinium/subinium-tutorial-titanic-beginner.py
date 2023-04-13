import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1.info()
print('-' * 20)
_input0.info()
_input1 = _input1.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
_input0 = _input0.drop(['Name', 'Ticket'], axis=1)
_input1['Pclass'].value_counts()
pclass_train_dummies = pd.get_dummies(_input1['Pclass'])
pclass_test_dummies = pd.get_dummies(_input0['Pclass'])
_input1 = _input1.drop(['Pclass'], axis=1, inplace=False)
_input0 = _input0.drop(['Pclass'], axis=1, inplace=False)
_input1 = _input1.join(pclass_train_dummies)
_input0 = _input0.join(pclass_test_dummies)
sex_train_dummies = pd.get_dummies(_input1['Sex'])
sex_test_dummies = pd.get_dummies(_input0['Sex'])
sex_train_dummies.columns = ['Female', 'Male']
sex_test_dummies.columns = ['Female', 'Male']
_input1 = _input1.drop(['Sex'], axis=1, inplace=False)
_input0 = _input0.drop(['Sex'], axis=1, inplace=False)
_input1 = _input1.join(sex_train_dummies)
_input0 = _input0.join(sex_test_dummies)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(0, inplace=False)
_input1 = _input1.drop(['Cabin'], axis=1)
_input0 = _input0.drop(['Cabin'], axis=1)
_input1['Embarked'].value_counts()
_input0['Embarked'].value_counts()
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input0['Embarked'] = _input0['Embarked'].fillna('S', inplace=False)
embarked_train_dummies = pd.get_dummies(_input1['Embarked'])
embarked_test_dummies = pd.get_dummies(_input0['Embarked'])
embarked_train_dummies.columns = ['S', 'C', 'Q']
embarked_test_dummies.columns = ['S', 'C', 'Q']
_input1 = _input1.drop(['Embarked'], axis=1, inplace=False)
_input0 = _input0.drop(['Embarked'], axis=1, inplace=False)
_input1 = _input1.join(embarked_train_dummies)
_input0 = _input0.join(embarked_test_dummies)
X_train = _input1.drop('Survived', axis=1)
Y_train = _input1['Survived']
X_test = _input0.drop('PassengerId', axis=1).copy()
logreg = LogisticRegression()