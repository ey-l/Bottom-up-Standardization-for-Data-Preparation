import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head(10)
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0.head(10)
_input1.shape
_input0.shape
_input1.info()
_input0.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.columns
_input1.head()
_input0.head()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input1.isnull().sum()
_input0.isnull().sum()
sns.catplot(x='Embarked', kind='count', data=_input1)
_input1['Embarked'] = _input1['Embarked'].fillna('S')
_input1.isnull().sum()
_input1['Cabin'] = _input1['Cabin'].fillna('Missing')
_input0['Cabin'] = _input0['Cabin'].fillna('Missing')
_input1.isnull().sum()
_input0.isnull().sum()
_input0['Fare'] = _input0['Fare'].median()
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = pd.get_dummies(_input1, columns=['Sex'], drop_first=True)
_input1 = pd.get_dummies(_input1, columns=['Embarked'], drop_first=True)
_input1['Fare'] = _input1['Fare'].astype(int)
_input1.loc[_input1.Fare <= 7.91, 'Fare'] = 0
_input1.loc[(_input1.Fare > 7.91) & (_input1.Fare <= 14.454), 'Fare'] = 1
_input1.loc[(_input1.Fare > 14.454) & (_input1.Fare <= 31), 'Fare'] = 2
_input1.loc[_input1.Fare > 31, 'Fare'] = 3
_input1['Age'] = _input1['Age'].astype(int)
_input1.loc[_input1['Age'] <= 16, 'Age'] = 0
_input1.loc[(_input1['Age'] > 16) & (_input1['Age'] <= 32), 'Age'] = 1
_input1.loc[(_input1['Age'] > 32) & (_input1['Age'] <= 48), 'Age'] = 2
_input1.loc[(_input1['Age'] > 48) & (_input1['Age'] <= 64), 'Age'] = 3
_input1.loc[_input1['Age'] > 64, 'Age'] = 4
_input0 = pd.get_dummies(_input0, columns=['Sex'], drop_first=True)
_input0 = pd.get_dummies(_input0, columns=['Embarked'], drop_first=True)
_input0['Fare'] = _input0['Fare'].astype(int)
_input0.loc[_input0.Fare <= 7.91, 'Fare'] = 0
_input0.loc[(_input0.Fare > 7.91) & (_input0.Fare <= 14.454), 'Fare'] = 1
_input0.loc[(_input0.Fare > 14.454) & (_input0.Fare <= 31), 'Fare'] = 2
_input0.loc[_input0.Fare > 31, 'Fare'] = 3
_input0['Age'] = _input0['Age'].astype(int)
_input0.loc[_input0['Age'] <= 16, 'Age'] = 0
_input0.loc[(_input0['Age'] > 16) & (_input0['Age'] <= 32), 'Age'] = 1
_input0.loc[(_input0['Age'] > 32) & (_input0['Age'] <= 48), 'Age'] = 2
_input0.loc[(_input0['Age'] > 48) & (_input0['Age'] <= 64), 'Age'] = 3
_input0.loc[_input0['Age'] > 64, 'Age'] = 4
_input1 = _input1.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=False)
_input1.describe()
_input1.Survived.value_counts() / len(_input1) * 100
_input1.groupby('Survived').mean()
_input1.groupby('Sex_male').mean()
_input1.corr()
plt.subplots(figsize=(10, 8))
sns.heatmap(_input1.corr(), annot=True, cmap='Blues_r')
plt.title('Correlation Among Variables', fontsize=20)
sns.barplot(x='Sex_male', y='Survived', data=_input1)
plt.title('Gender Distribution - Survived', fontsize=16)
sns.barplot(x='Pclass', y='Survived', data=_input1)
plt.title('Passenger Class Distribution - Survived', fontsize=16)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X = _input1.drop(['Survived'], axis=1)
y = _input1['Survived']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.22, random_state=5)
print(len(X_train), len(X_test), len(y_train), len(y_test))
logReg = LogisticRegression()