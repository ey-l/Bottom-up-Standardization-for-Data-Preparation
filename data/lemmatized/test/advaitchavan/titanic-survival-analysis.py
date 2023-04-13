import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, svm
from sklearn.metrics import accuracy_score
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1
print('The shape of our training set: %s passengers and %s features' % (_input1.shape[0], _input1.shape[1]))
_input1.info()
_input1.isnull().sum()
sns.heatmap(_input1[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True)
sns.set(rc={'figure.figsize': (12, 10)})
_input1['SibSp'].unique()
sns.catplot(x='SibSp', y='Survived', data=_input1, kind='bar', height=8)
sns.FacetGrid(_input1, col='Survived', height=7).map(sns.distplot, 'Age').set_ylabels('Survival Probability')
sns.barplot(x='Sex', y='Survived', data=_input1)
sns.catplot(x='Pclass', y='Survived', data=_input1, kind='bar', height=6)
(_input1['Embarked'].value_counts(), _input1['Embarked'].isnull().sum())
_input1['Embarked'] = _input1['Embarked'].fillna('S')
_input1.isnull().sum()
sns.catplot(x='Embarked', y='Survived', data=_input1, height=5, kind='bar')
sns.catplot(x='Pclass', col='Embarked', data=_input1, kind='count', height=7)
mean_age = _input1['Age'].mean()
std_age = _input1['Age'].std()
(mean_age, std_age)
random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size=177)
age_slice = _input1['Age'].copy()
age_slice[np.isnan(age_slice)] = random_age
_input1['Age'] = age_slice
_input1.isnull().sum()
list_column_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
_input1 = _input1.drop(list_column_to_drop, axis=1, inplace=False)
_input1.head(10)
genders = {'male': 0, 'female': 1}
_input1['Sex'] = _input1['Sex'].map(genders)
ports = {'S': 0, 'C': 1, 'Q': 2}
_input1['Embarked'] = _input1['Embarked'].map(ports)
_input1.head()
df_train_x = _input1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df_train_y = _input1[['Survived']]
(x_train, x_test, y_train, y_test) = train_test_split(df_train_x, df_train_y, test_size=0.2, random_state=18)
clf1 = RandomForestClassifier()