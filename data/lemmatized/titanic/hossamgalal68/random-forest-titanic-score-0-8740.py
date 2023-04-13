import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer as si
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
all_data = pd.concat([_input1, _input0])
print('All data shape is ', all_data.shape)
print('Training data shape is ', _input1.shape)
print('Test data shape is ', _input0.shape)
print('Submission data shape is ', _input2.shape)
_input1
_input1.info()
_input1.describe()
_input0.info()
_input0.describe()
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input1 = _input1.drop(['Cabin', 'Ticket', 'Name'], 1)
_input1 = _input1.dropna(axis=0, subset=['Embarked'])
_input1 = _input1.fillna(0, inplace=False)
_input1
_input1.isnull().sum()
_input0
_input0.isnull().sum()
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean())
_input0 = _input0.drop(['Cabin', 'Ticket', 'Name'], 1)
_input0 = _input0.fillna(0, inplace=False)
_input0.isnull().sum()
_input0
training_encoding = _input1.copy()
training_encoding
training_encoding.info()
training_encoding['Sex'].value_counts()
training_encoding['Sex'] = training_encoding['Sex'].factorize(['female', 'male'])[0]
training_encoding['Sex'].value_counts()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
training_encoding['Embarked_N'] = label_encoder.fit_transform(training_encoding['Embarked'])
training_encoding
training_encoding = training_encoding.drop('Embarked', inplace=False, axis=1)
training_encoding
training_encoding.info()
test_encoding = _input0.copy()
test_encoding.info()
test_encoding['Sex'].value_counts()
test_encoding['Sex'] = test_encoding['Sex'].factorize(['female', 'male'])[0]
test_encoding['Sex'].value_counts()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
test_encoding['Embarked_N'] = label_encoder.fit_transform(test_encoding['Embarked'])
test_encoding
test_encoding = test_encoding.drop('Embarked', inplace=False, axis=1)
test_encoding
test_encoding.info()
df_test = test_encoding.copy()
df_test['Survived'] = _input2['Survived']
df_test
df_test.describe()
df_test.isnull().sum()
df_training = training_encoding.copy()
df_training
plt.figure(figsize=(30, 30))
sns.heatmap(df_training.corr(), annot=True, cmap='mako', annot_kws={'size': 14})
fig = plt.figure(figsize=(5, 5))
df_training['Survived'].value_counts().plot(kind='pie', autopct='%.1f%%')
plt.ylabel(' ', fontsize=15)
plt.title('Survived')
print('')
sns.countplot(data=df_training, x='Survived', hue='Pclass').set(xticklabels=['Did not Survive', 'Survied'], title='Titanic Survival Data')
sns.countplot(data=df_training, x='Survived', hue='Sex').set(xticklabels=['Did not Survive', 'Survied'], title='Titanic Survival Data')
sns.distplot(df_training['Age'])
fig = plt.figure(figsize=(5, 5))
df_training['Embarked_N'].value_counts().plot(kind='pie', autopct='%.1f%%')
plt.ylabel(' ', fontsize=15)
plt.title('Embarked_N')
print('')
sns.countplot(data=df_training, x='Survived', hue='SibSp').set(xticklabels=['Did not Survive', 'Survied'], title='Titanic Survival Data')
sns.countplot(data=df_training, x='Survived', hue='Parch').set(xticklabels=['Did not Survive', 'Survied'], title='Titanic Survival Data')
train_split = df_training.copy()
test_split = df_test.copy()
train_split
test_split
train_split.columns
test_split.columns
train_split = train_split[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_N', 'Survived']]
test_split = test_split[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_N', 'Survived']]
x_train = train_split.drop(['Survived'], axis=1).values
y_train = train_split['Survived'].values
x_test = test_split.drop(['Survived'], axis=1).values
y_test = test_split['Survived'].values
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
scalar = StandardScaler()
x_test = scalar.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=20)