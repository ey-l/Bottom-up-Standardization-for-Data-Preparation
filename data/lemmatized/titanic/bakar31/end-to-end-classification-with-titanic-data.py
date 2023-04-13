import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
(_input1.shape, _input0.shape)
_input1 = _input1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
_input1.isna().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1.Embarked.value_counts()
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input1.isna().sum()
_input1.Survived.value_counts()
_input1.Survived.value_counts().plot(kind='bar', color=['lightblue', 'lightgreen'])
_input1.Sex.value_counts()
_input1.Sex.value_counts().plot(kind='bar', color=['skyblue', 'plum'])
pd.crosstab(_input1.Sex, _input1.Survived)
pd.crosstab(_input1.Sex, _input1.Survived).plot(kind='bar', color=['slategray', 'salmon'])
pd.crosstab(_input1.Pclass, _input1.Survived)
pd.crosstab(_input1.Pclass, _input1.Survived).plot(kind='bar', color=['slategray', 'lightcoral'])
_input1.Embarked.value_counts()
sns.countplot(x='Embarked', data=_input1)
sns.displot(x='Age', data=_input1, color='cadetblue', kde=True)
sns.displot(x='Fare', data=_input1, kind='kde')
sns.lmplot(x='Age', y='Survived', hue='Pclass', data=_input1)
correlation_matrix = _input1.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='YlGnBu')
_input1['family'] = _input1['SibSp'] + _input1['Parch']
_input1.head(10)
_input1['Age'] = np.log(_input1['Age'] + 1)
_input1['Age'].plot(kind='density', figsize=(10, 6))
_input1['Fare'] = np.log(_input1['Fare'] + 1)
_input1['Fare'].plot(kind='density', figsize=(10, 6))
_input1.head(10)
x = _input1.drop('Survived', axis=1)
y = _input1['Survived']
x.shape
x.head()
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()
transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder='passthrough')
encoded = transformer.fit_transform(x)
encoded_df = pd.DataFrame(encoded)
encoded_df.shape
encoded_df.head()
encoded_x = encoded_df.drop([0, 2, 5], axis=1)
encoded_x.head()
encoded_x.shape
y.shape
_input0['family'] = _input0['SibSp'] + _input0['Parch']
_input0.head()
_input0['Age'] = np.log(_input0['Age'] + 1)
_input0['Fare'] = np.log(_input0['Fare'] + 1)
_input0['Age'].plot(kind='density', figsize=(10, 6))
_input0['Fare'].plot(kind='density', figsize=(10, 6))
_input0.head(10)
_input0 = _input0.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
_input0.head(10)
_input0.isna().sum()
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].mean(), inplace=False)
_input0.isna().sum()
categorical_features = ['Sex', 'Embarked', 'Pclass']
onehotencode = OneHotEncoder()
transformer = ColumnTransformer([('Encoder', onehotencode, categorical_features)], remainder='passthrough')
encoded_test = transformer.fit_transform(_input0)
encoded_test = pd.DataFrame(encoded_test)
encoded_test.head()
encoded_test_x = encoded_test.drop([0, 2, 5], axis=1)
encoded_test_x.head()
encoded_test_x.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(encoded_x, y, random_state=31)
(len(x_train), len(x_test), len(y_train), len(y_test))
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter=1000, random_state=4)