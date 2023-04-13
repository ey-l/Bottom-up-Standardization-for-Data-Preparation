import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head(5)
sns.countplot(x=_input1['Sex'], hue=_input1['Survived'])
sns.countplot(x=_input1['Pclass'], hue=_input1['Survived'])
train_data1 = _input1.drop(['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch'], axis=1)
train_data1['Embarked'] = train_data1['Embarked'].fillna('C')
train_data1['Sex'] = train_data1['Sex'].replace('female', 0, inplace=False)
train_data1['Sex'] = train_data1['Sex'].replace('male', 1, inplace=False)
train_data1['Embarked'] = train_data1['Embarked'].replace('S', 0, inplace=False)
train_data1['Embarked'] = train_data1['Embarked'].replace('C', 1, inplace=False)
train_data1['Embarked'] = train_data1['Embarked'].replace('Q', 2, inplace=False)
train_data1['Age'] = train_data1['Age'].fillna('39', inplace=False)
train_data1.head(5)
train_data1 = train_data1.dropna(inplace=False)
train_data1.isnull().sum()
train_data1.info()
x_train = train_data1.drop('Survived', axis=1)
y_train = train_data1['Survived']
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0.info()
test_data1 = _input0.drop(['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch'], axis=1)
test_data1.info()
test_data1['Age'] = test_data1['Age'].fillna('34', inplace=False)
test_data1.info()
test_data1['Fare'] = test_data1['Fare'].fillna('7.75', inplace=False)
test_data1.info()
test_data1['Sex'] = test_data1['Sex'].replace('female', 0, inplace=False)
test_data1['Sex'] = test_data1['Sex'].replace('male', 1, inplace=False)
test_data1['Embarked'] = test_data1['Embarked'].replace('S', 0, inplace=False)
test_data1['Embarked'] = test_data1['Embarked'].replace('C', 1, inplace=False)
test_data1['Embarked'] = test_data1['Embarked'].replace('Q', 2, inplace=False)
x_test = test_data1
log = LogisticRegression()