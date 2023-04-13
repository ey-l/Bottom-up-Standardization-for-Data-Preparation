import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1.isna().sum()
_input1.dtypes
sns.histplot(data=_input1, x='Survived', color='r')
sns.barplot(data=_input1, x='Survived', y='Age')
sns.boxplot(data=_input1, y='Age', x='Survived')
sns.barplot(data=_input1, x='Survived', y='Fare')
sns.boxplot(data=_input1, y='Fare', x='Survived')
colunas = ['Pclass', 'SibSp', 'Parch', 'Fare']
X = _input1[colunas]
y = _input1.Survived
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=0)
model = DecisionTreeClassifier(random_state=1)