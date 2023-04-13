import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.nunique().sum
_input1.isnull().sum()
_input1 = _input1.drop(columns='Cabin', axis=1)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1.Embarked.mode()
_input1['Embarked'] = _input1['Embarked'].fillna('S')
_input1.isnull().sum()
_input1.describe()
_input1['Survived'].value_counts()
sns.countplot('Survived', data=_input1)
_input1['Sex'].value_counts()
sns.countplot('Sex', data=_input1)
sns.countplot('Sex', hue='Survived', data=_input1)
sns.countplot('Pclass', data=_input1)
sns.countplot('Pclass', hue='Survived', data=_input1)
_input1['Sex'].value_counts()
_input1['Embarked'].value_counts()
_input1 = _input1.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=False)
_input1.head()
X = _input1.drop(columns=['Name', 'Ticket', 'Survived'], axis=1)
y = _input1['Survived']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
lr_model = LogisticRegression()
rfc_model = RandomForestClassifier()
knn_model = KNeighborsClassifier(n_neighbors=15)