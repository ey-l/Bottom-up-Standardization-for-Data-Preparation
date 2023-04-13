import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import xgboost
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1
sns.countplot(x='Sex', data=_input1)
sns.countplot(x='Sex', hue='Survived', data=_input1)
sns.distplot(x=_input1.Age, hist=True)
sns.distplot(x=_input1.Fare, hist=True)
sns.countplot(x='Pclass', hue='Survived', data=_input1)
sns.countplot(x='Survived', hue='Pclass', data=_input1)
sns.countplot(x='Embarked', data=_input1)
scatter_matrix(_input1)
print(_input1.corr())
sns.heatmap(_input1.corr())
_input1.isnull()
_input1.info()
_input1.describe()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna(_input1['Embarked'].mode()[0], inplace=False)
_input1.info()
data = _input1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data
dum = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked'])
print(dum.corr())
upd_dum = dum.drop(['Pclass_2', 'Sex_male', 'Embarked_Q'], axis=1)
upd_dum.info()
sns.heatmap(upd_dum.isnull(), cbar=False)
upd_dum
X = upd_dum.drop(['Survived'], axis=1)
y = upd_dum['Survived']
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestClassifier(n_estimators=65), X, y, cv=3).mean()
cross_val_score(LogisticRegression(), X, y, cv=3).mean()
cross_val_score(GaussianNB(), X, y, cv=3).mean()
cross_val_score(MultinomialNB(), X, y, cv=3).mean()
cross_val_score(xgboost.XGBClassifier(n_estimators=20), X, y, cv=3).mean()
model = xgboost.XGBClassifier(n_estimators=20)