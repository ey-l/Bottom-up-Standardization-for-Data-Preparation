import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/titanic/test.csv', index_col='PassengerId')
print(_input1.shape)
print(_input0.shape)
_input1.head()
print(_input1.isnull().sum())
print('')
print(_input0.isnull().sum())
y = _input1.Survived
_input1 = _input1.drop(['Survived'], axis=1, inplace=False)
_input1['Title'] = 0
_input1['Title'] = _input1.Name.str.extract('([A-Za-z]+)\\.')
pd.crosstab(_input1.Title, _input1.Sex).T
_input1['Title'] = _input1['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'], ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Master', 'Miss', 'Rare', 'Rare', 'Mr', 'Mrs', 'Rare', 'Rare', 'Rare'], inplace=False)
_input1.groupby('Title')['Age'].median()
sns.countplot(x='Title', hue='Survived', data=pd.concat([_input1, y], axis=1), palette='Blues_d')
_input1.loc[_input1.Age.isnull() & (_input1.Title == 'Master'), 'Age'] = 3.5
_input1.loc[_input1.Age.isnull() & (_input1.Title == 'Miss'), 'Age'] = 21
_input1.loc[_input1.Age.isnull() & (_input1.Title == 'Mr'), 'Age'] = 30
_input1.loc[_input1.Age.isnull() & (_input1.Title == 'Mrs'), 'Age'] = 34.5
_input1.loc[_input1.Age.isnull() & (_input1.Title == 'Rare'), 'Age'] = 44.5
_input1.Age.isnull().sum()
_input0['Title'] = 0
_input0['Title'] = _input0.Name.str.extract('([A-Za-z]+)\\.')
_input0['Title'] = _input0['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'Dona'], ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Master', 'Miss', 'Rare', 'Rare', 'Mr', 'Mrs', 'Rare', 'Rare', 'Rare', 'Rare'], inplace=False)
_input0.loc[_input0.Age.isnull() & (_input0.Title == 'Master'), 'Age'] = 3.5
_input0.loc[_input0.Age.isnull() & (_input0.Title == 'Miss'), 'Age'] = 21
_input0.loc[_input0.Age.isnull() & (_input0.Title == 'Mr'), 'Age'] = 30
_input0.loc[_input0.Age.isnull() & (_input0.Title == 'Mrs'), 'Age'] = 34.5
_input0.loc[_input0.Age.isnull() & (_input0.Title == 'Rare'), 'Age'] = 44.5
_input0.Age.isnull().sum()
_input1['HasCabin'] = _input1['Cabin'].notnull()
_input0['HasCabin'] = _input0['Cabin'].notnull()
sns.countplot(x='HasCabin', hue='Survived', data=pd.concat([_input1, y], axis=1), palette='Blues_d')
categorical_cols = ['Pclass', 'Sex', 'Embarked', 'HasCabin']
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
my_cols = categorical_cols + numerical_cols
X_train = _input1[my_cols].copy()
X_test = _input0[my_cols].copy()
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train = my_pipeline.fit_transform(X_train)
grid = {'n_estimators': [100, 125, 150, 175, 200, 225, 250], 'max_depth': [4, 6, 8, 10, 12]}
clf = RandomForestClassifier(random_state=0)
grid_model = GridSearchCV(clf, grid, cv=4)