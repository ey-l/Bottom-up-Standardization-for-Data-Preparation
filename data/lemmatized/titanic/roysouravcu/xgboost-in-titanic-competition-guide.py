import math, time, random, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
_input1.isnull().sum()
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=_input1)
print(_input1.Survived.value_counts())
sns.set_style('whitegrid')
sns.countplot(x='Pclass', data=_input1)
print(_input1.Pclass.value_counts())
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=_input1, palette='rainbow')
sns.set_style('darkgrid')
sns.countplot(x='Sex', data=_input1)
print(_input1.Sex.value_counts())
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=_input1, palette='rainbow')
pd.set_option('display.max_rows', 88)
_input1.Age.value_counts()
_input1.Age.isnull().sum()
sns.set_style('darkgrid')
sns.histplot(_input1['Age'].dropna(), bins=40, color='blue', kde=True)
_input1['Age'].hist(bins=40, color='darkred', alpha=0.85)
_input1.describe()
_input1.info()
sns.set_style('whitegrid')
sns.boxplot(x='Pclass', y='Age', data=_input1, palette='rainbow')
sns.set_style('whitegrid')
sns.countplot(y='SibSp', data=_input1)
print(_input1.SibSp.value_counts())
sns.set_style('whitegrid')
sns.countplot(x='Parch', data=_input1, palette='rainbow')
print(_input1.Parch.value_counts())
_input1.Ticket.value_counts()
_input1.Ticket.unique()
print(len(_input1.Fare.unique()))
print(_input1.Fare.isnull().sum())
_input1.Cabin.isnull().sum()
print(_input1.Embarked.value_counts())
sns.countplot(y='Embarked', data=_input1)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
_input1 = _input1.drop('Cabin', axis=1)
_input1.head()
_input1.isnull().sum()
_input1['Embarked'] = _input1['Embarked'].fillna(_input1['Embarked'].value_counts().index[0])
_input1.isnull().sum()
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = _input1[features]
y = _input1['Survived']
X.isnull().sum()
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X['Sex'] = LE.fit_transform(X['Sex'])
X['Embarked'] = LE.fit_transform(X['Embarked'])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)