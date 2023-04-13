import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.info()
_input1.isnull().sum()
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna(_input1['Embarked'].value_counts().idxmax(), inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(skipna=True), inplace=False)
_input1.isnull().sum()
_input1.head()
_input1['Sex'] = _input1['Sex'].replace('female', 0, inplace=False)
_input1['Sex'] = _input1['Sex'].replace('male', 1, inplace=False)
_input1['Embarked'] = _input1['Embarked'].replace('S', 0, inplace=False)
_input1['Embarked'] = _input1['Embarked'].replace('C', 1, inplace=False)
_input1['Embarked'] = _input1['Embarked'].replace('Q', 2, inplace=False)
_input1.dtypes
_input0.isnull().sum()
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(skipna=True), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].median(skipna=True), inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input0['Sex'] = _input0['Sex'].replace('female', 0, inplace=False)
_input0['Sex'] = _input0['Sex'].replace('male', 1, inplace=False)
_input0['Embarked'] = _input0['Embarked'].replace('S', 0, inplace=False)
_input0['Embarked'] = _input0['Embarked'].replace('C', 1, inplace=False)
_input0['Embarked'] = _input0['Embarked'].replace('Q', 2, inplace=False)
_input0.dtypes
_input1.shape
_input0.shape
_input1.head()
outcome_data = _input1['Survived']
_input1 = _input1.drop(['Survived', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=False)
_input0 = _input0.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=False)
from sklearn.model_selection import train_test_split
X = _input1.values
y = outcome_data.values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=4, max_features='auto', max_leaf_nodes=5, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=15, min_weight_fraction_leaf=0.0, n_estimators=350, n_jobs=None, oob_score=False, random_state=1, verbose=0, warm_start=False)