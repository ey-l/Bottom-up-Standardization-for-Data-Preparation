import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
_input1 = pd.read_csv('data/input/titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/titanic/test.csv', index_col='PassengerId')
_input1.head()
_input0.head()
_input1.shape
_input0.shape
_input1.columns
_input0.columns
_input1.isna().sum()
_input0.isna().sum()
_input1.nunique()
_input0.nunique()
_input1.info()
_input0.info()
_input1.describe()
_input1.describe(include=['O'])
_input1['Cabin'].unique()
_input1['Cabin'] = _input1['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'na')
_input1['Cabin'].unique()
_input0['Cabin'] = _input0['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'na')
_input0['Cabin'].unique()
_input1 = _input1.reset_index(inplace=False)
_input0 = _input0.reset_index(inplace=False)
_input1 = _input1.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=False)
test_passenger_ids = _input0['PassengerId']
_input0 = _input0.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(skipna=True), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(skipna=True), inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input0['Embarked'] = _input0['Embarked'].fillna('S', inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].mean(skipna=True), inplace=False)
_input1.isna().sum()
_input0.isna().sum()
sex = {'male': 0, 'female': 1}
_input1['Sex'] = [sex[i] for i in _input1['Sex']]
_input0['Sex'] = [sex[i] for i in _input0['Sex']]
embarked = {'S': 0, 'C': 1, 'Q': 2}
_input1['Embarked'] = [embarked[i] for i in _input1['Embarked']]
_input0['Embarked'] = [embarked[i] for i in _input0['Embarked']]
cabin_plot = _input1[['Cabin', 'Survived']]
_input1['Cabin'] = LabelEncoder().fit_transform(_input1['Cabin'])
_input0['Cabin'] = LabelEncoder().fit_transform(_input0['Cabin'])
plt.figure(figsize=(20, 10))
sns.histplot(x='Age', data=_input1)
plt.title('Histogram (Age)')
plt.figure(figsize=(20, 10))
sns.kdeplot(x='Age', data=_input1, fill=True)
plt.title('KDE (Age)')
plt.figure(figsize=(20, 2))
sns.boxplot(x='Age', data=_input1)
plt.title('Boxplot (Age)')
plt.figure(figsize=(20, 2))
sns.violinplot(x='Age', data=_input1)
plt.title('Violin plot (Age)')
plt.figure(figsize=(20, 10))
sns.histplot(x='Fare', data=_input1)
plt.title('Histogram (Fare)')
plt.figure(figsize=(20, 10))
sns.kdeplot(x='Fare', data=_input1, fill=True)
plt.title('KDE (Fare)')
plt.figure(figsize=(20, 2))
sns.boxplot(x='Fare', data=_input1)
plt.title('Boxplot (Fare)')
plt.figure(figsize=(20, 2))
sns.violinplot(x='Fare', data=_input1)
plt.title('Violin plot (Fare)')
plt.figure(figsize=(10, 5))
sns.countplot(x='Sex', data=_input1)
plt.title('Countplot (Sex)')
plt.xticks([0, 1], ['Male', 'Female'])
plt.figure(figsize=(20, 10))
sns.countplot(x='Pclass', hue='Survived', data=_input1)
plt.title('Survived VS Pclass')
plt.figure(figsize=(10, 5))
_input1.loc[_input1['Sex'] == 0, 'Survived'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Percentage survived (Male)')
plt.figure(figsize=(10, 5))
_input1.loc[_input1['Sex'] == 1, 'Survived'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Percentage survived (Female)')
plt.figure(figsize=(20, 10))
sns.countplot(x='SibSp', hue='Survived', data=_input1)
plt.title('Survived VS SibSp')
plt.figure(figsize=(20, 10))
sns.countplot(x='Cabin', hue='Survived', data=cabin_plot)
plt.title('Survived VS Cabin')
plt.figure(figsize=(20, 10))
sns.kdeplot(x='Age', hue='Survived', data=_input1, shade=True)
plt.title('Survived VS Age')
plt.figure(figsize=(20, 10))
sns.kdeplot(x='Fare', hue='Survived', data=_input1, shade=True)
plt.title('Survived VS Fare')
(fig, axes) = plt.subplots(3, 2, figsize=(20, 20))
sns.violinplot(x='Pclass', y='Age', hue='Survived', split=True, data=_input1, ax=axes[0, 0])
sns.violinplot(x='Sex', y='Age', hue='Survived', split=True, data=_input1, ax=axes[0, 1])
sns.violinplot(x='SibSp', y='Age', hue='Survived', split=True, data=_input1, ax=axes[1, 0])
sns.violinplot(x='Parch', y='Age', hue='Survived', split=True, data=_input1, ax=axes[1, 1])
sns.violinplot(x='Embarked', y='Age', hue='Survived', split=True, data=_input1, ax=axes[2, 0])
sns.violinplot(x='Cabin', y='Age', hue='Survived', split=True, data=_input1, ax=axes[2, 1])
y = _input1['Survived']
X = _input1.drop('Survived', axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
decision_tree_model = DecisionTreeClassifier(random_state=2)