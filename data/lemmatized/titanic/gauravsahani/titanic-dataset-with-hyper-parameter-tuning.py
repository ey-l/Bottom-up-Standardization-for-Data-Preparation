import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input0.head()
_input1.isnull().sum()
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input1 = _input1.drop(['Ticket'], axis=1, inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mode()[0])
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False)
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Ticket'], axis=1, inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mode()[0])
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].mode()[0])
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False)
_input0.isnull().sum()
_input1.describe()
_input1.head()
S_Dummy = pd.get_dummies(_input1['Sex'], drop_first=True)
Embarked_Dummy = pd.get_dummies(_input1['Embarked'], drop_first=True)
_input1 = pd.concat([_input1, S_Dummy, Embarked_Dummy], axis=1)
_input1
_input1 = _input1.drop(['Sex', 'Embarked'], axis=1, inplace=False)
_input1.head()
from sklearn.model_selection import train_test_split
X = _input1[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
y = _input1[['Survived']]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()