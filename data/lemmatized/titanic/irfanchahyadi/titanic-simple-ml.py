import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (16, 10)
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1['dataset'] = 'train'
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0['dataset'] = 'test'
df = pd.concat([_input1, _input0], sort=True, copy=False)
_input1.info()
_input1.nunique().sort_values()
num_features = _input1.select_dtypes(['float64', 'int64']).columns.tolist()
cat_features = _input1.select_dtypes(['object']).columns.tolist()
print('{} numerical features:\n{} \nand {} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))
num_features.remove('PassengerId')
num_features = sorted(num_features)
num_features
_input1[num_features].describe()
print('{:.2f}% survival rate, {} out of {} survived'.format(_input1.Survived.sum() / len(_input1) * 100, _input1.Survived.sum(), len(_input1)))
corrplot = sns.heatmap(_input1[num_features].corr(), cmap=plt.cm.Reds, annot=True)
abs(_input1[num_features].corr()['Survived']).sort_values(ascending=False)
g = sns.FacetGrid(_input1, col='Survived')
g.map(sns.distplot, 'Pclass')
_input1.groupby('Pclass').agg(['mean', 'count'])['Survived']
sns.boxplot(data=df, x='Fare', y='Pclass', orient='h')
age_plot = sns.distplot(_input1[_input1.Age.notnull()].Age)
_input1.Age.isnull().sum()
df['Age'] = df[['Age']].applymap(lambda x: df.Age.mean() if pd.isnull(x) else x)
print(_input1.groupby('SibSp').agg(['mean', 'count'])['Survived'])
print(_input1.groupby('Parch').agg(['mean', 'count'])['Survived'])
df['Family'] = df.SibSp + df.Parch
print(df[df.dataset == 'train'].groupby('Family').agg(['mean', 'count'])['Survived'])
df = df.drop(['SibSp', 'Parch'], axis=1, inplace=False)
fare_plot = sns.distplot(_input1.Fare)
_input1['Fare_std'] = _input1[['Fare']].apply(lambda x: abs(x - x.mean()) / x.std())
_input1[['Fare', 'Fare_std']].sort_values('Fare_std', ascending=False).head(25)
df = df[(df.dataset != 'train') | (df.Fare < 200)]
_input1[['Name', 'Ticket']].head(20)
df = df.drop(['Name', 'Ticket'], axis=1, inplace=False)
g = sns.FacetGrid(_input1, col='Survived').map(sns.countplot, 'Sex')
_input1.groupby('Sex').agg(['mean', 'count'])['Survived']
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
print(_input1.Cabin.isnull().sum())
print(_input1.Cabin.nunique())
_input1.Cabin[_input1.Cabin.notnull()].head(10)
df['Cabin'] = df[['Cabin']].applymap(lambda x: 'Z' if pd.isnull(x) else x[0])
pivoted = df.groupby(['Cabin', 'Survived']).size().reset_index().pivot(index='Cabin', columns='Survived')
stackedplot = pivoted.plot.bar(stacked=True)
stackedplot_withoutZ = pivoted.drop('Z').plot.bar(stacked=True)
df = pd.get_dummies(df, columns=['Cabin'], prefix='Cabin')
df.head()
pivoted = df.groupby(['Embarked', 'Survived']).size().reset_index().pivot(index='Embarked', columns='Survived')
stackedplot = pivoted.plot.bar(stacked=True)
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
df.head()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
X = df[df.dataset == 'train'].drop(['PassengerId', 'dataset', 'Survived'], axis=1)
y = df[df.dataset == 'train']['Survived']
X_test = df[df.dataset == 'test'].drop(['PassengerId', 'dataset', 'Survived'], axis=1)
(X_train, X_validate, y_train, y_validate) = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=1000)