import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1.shape
sns.countplot(x='Survived', data=_input1)
sns.countplot(x='Survived', hue='Sex', data=_input1, palette='ocean')
sns.countplot(x='Survived', hue='Pclass', data=_input1, palette='flare')
_input1['Age'].plot.hist()
_input1['Fare'].plot.hist(bins=20, figsize=(10, 5))
sns.countplot(x='SibSp', data=_input1, palette='rocket')
_input1['Parch'].plot.hist()
sns.countplot(x='Parch', data=_input1, palette='magma')
_input1.head()
_input1 = _input1.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=False)
_input1.head()
_input1.shape
_input1 = pd.get_dummies(_input1)
_input1.head()
_input1['Age'] = _input1['Age'] / max(_input1['Age'])
_input1['Fare'] = _input1['Fare'] / max(_input1['Fare'])
_input1.head()
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input1.isnull().sum()
_input1.head()
_input1.shape
x = _input1.drop('Survived', axis=1)
y = _input1['Survived']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=30, random_state=42)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()