import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
sns.set(style='white', context='notebook', palette='deep')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
PAS = list(_input0.PassengerId)
_input1.info()
_input1.describe().T
_input1.head(10)
_input1.info()
print('_' * 40)
_input0.info()
report = ProfileReport(_input0)
report
sns.pairplot(_input1, hue='Survived', palette='Paired')
sns.heatmap(_input1.corr(), annot=True, cmap='YlGnBu', linewidths=1)
sns.countplot(x=_input1['Sex'], hue=_input1['Survived'])
_input1[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x=_input1['Parch'], hue=_input1['Survived'])
sns.countplot(x=_input1['SibSp'], hue=_input1['Survived'])
g = sns.kdeplot(_input1['Age'][(_input1['Survived'] == 0) & _input1['Age'].notnull()], color='Red', shade=True)
g = sns.kdeplot(_input1['Age'][(_input1['Survived'] == 1) & _input1['Age'].notnull()], ax=g, color='Blue', shade=True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g = g.legend(['Not Survived', 'Survived'])
_input1[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Embarked', data=_input1, palette='rainbow')
_input1['Cabin'] = _input1['Cabin'].apply(lambda i: i[0] if pd.notnull(i) else 'Z')
_input0['Cabin'] = _input0['Cabin'].apply(lambda i: i[0] if pd.notnull(i) else 'Z')
print(_input1['Cabin'].value_counts(), '\n--------\n', _input0['Cabin'].value_counts())
_input1.loc[339, 'Cabin'] = 'A'
_input1['Cabin'].unique()
_input1[_input1['Cabin'] == 'T'].index.values
_input1['Cabin'] = _input1['Cabin'].replace(['A', 'B', 'C'], 'ABC')
_input1['Cabin'] = _input1['Cabin'].replace(['D', 'E'], 'DE')
_input1['Cabin'] = _input1['Cabin'].replace(['F', 'G'], 'FG')
_input0['Cabin'] = _input0['Cabin'].replace(['A', 'B', 'C'], 'ABC')
_input0['Cabin'] = _input0['Cabin'].replace(['D', 'E'], 'DE')
_input0['Cabin'] = _input0['Cabin'].replace(['F', 'G'], 'FG')
_input1['Cabin'].unique()
_input1.head()
_input1 = _input1.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=False)
_input0 = _input0.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(skipna=True), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(skipna=True), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].median(skipna=True), inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input0['Embarked'] = _input0['Embarked'].fillna('S', inplace=False)
_input1.groupby('Embarked').mean()
_input1.groupby('Cabin').mean()
gender = {'male': 0, 'female': 1}
_input1.Sex = [gender[item] for item in _input1.Sex]
_input0.Sex = [gender[item] for item in _input0.Sex]
embarked = {'S': 0, 'Q': 1, 'C': 2}
_input1.Embarked = [embarked[item] for item in _input1.Embarked]
_input0.Embarked = [embarked[item] for item in _input0.Embarked]
Cabins = {'Z': 0, 'FG': 1, 'ABC': 2, 'DE': 3}
_input1.Cabin = [Cabins[item] for item in _input1.Cabin]
_input0.Cabin = [Cabins[item] for item in _input0.Cabin]
mask = np.triu(np.ones_like(_input1.corr(), dtype=bool))
(fig, ax) = plt.subplots(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(_input1.corr(), mask=mask, cmap='YlGnBu', vmax=0.3, center=0, annot=True, square=True)
expected_values = _input1['Survived']
_input1 = _input1.drop('Survived', axis=1, inplace=False)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
minmax = MinMaxScaler()