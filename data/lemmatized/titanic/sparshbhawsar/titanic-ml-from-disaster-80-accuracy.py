import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, precision_score, f1_score, roc_auc_score, accuracy_score
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1 = _input1.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=False)
print(_input1.shape)
_input1.head()
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0 = _input0.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
print(_input0.shape)
_input0.head()
sex_mapping = {'male': 0, 'female': 1}
_input1.Sex = _input1.Sex.map(sex_mapping)
_input0.Sex = _input0.Sex.map(sex_mapping)
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
_input1.Embarked = _input1.Embarked.map(embarked_mapping)
_input0.Embarked = _input0.Embarked.map(embarked_mapping)
_input1['Embarked'] = _input1['Embarked'].fillna(value=_input1['Embarked'].median(), inplace=False)
_input0['Embarked'] = _input0['Embarked'].fillna(value=_input1['Embarked'].median(), inplace=False)
_input1['Fare'] = _input1['Fare'].fillna(value=_input1['Fare'].median(), inplace=False)
_input0['Fare'] = _input0['Fare'].fillna(value=_input1['Fare'].median(), inplace=False)
_input1['FamilySize'] = _input1['SibSp'] + _input1['Parch'] + 1
_input0['FamilySize'] = _input0['SibSp'] + _input0['Parch'] + 1
_input1 = pd.get_dummies(_input1)
_input0 = pd.get_dummies(_input0)
_input1.isnull().sum()
_input0.isnull().sum()
useless_features = ['Survived', 'PassengerId']
useful_features = [i for i in _input1.columns if i not in useless_features]
imputer = IterativeImputer(max_iter=25, random_state=42)
train_data_imptr = imputer.fit_transform(_input1[useful_features])
train_data_imtr = pd.DataFrame(train_data_imptr, columns=useful_features)
_input1 = _input1.drop(useful_features, axis=1)
_input1 = pd.concat([_input1, train_data_imtr], axis=1)
test_data_imptr = imputer.transform(_input0[useful_features])
test_data_imtr = pd.DataFrame(test_data_imptr, columns=useful_features)
_input0 = _input0.drop(useful_features, axis=1)
_input0 = pd.concat([_input0, test_data_imtr], axis=1)
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Survived'] = _input1['Survived'].astype(int)
_input1['Pclass'] = _input1['Pclass'].astype(int)
_input1['SibSp'] = _input1['SibSp'].astype(int)
_input1['Parch'] = _input1['Parch'].astype(int)
_input1['Embarked'] = _input1['Embarked'].astype(int)
_input1['FamilySize'] = _input1['FamilySize'].astype(int)
_input0['PassengerId'] = _input0['PassengerId'].astype(int)
_input0['Pclass'] = _input0['Pclass'].astype(int)
_input0['SibSp'] = _input0['SibSp'].astype(int)
_input0['Parch'] = _input0['Parch'].astype(int)
_input0['Embarked'] = _input0['Embarked'].astype(int)
_input0['FamilySize'] = _input0['FamilySize'].astype(int)
(_input1.shape, _input0.shape)
_input1['Survived'].hist()
_input1.head()
y = _input1['Survived']
X = _input1.drop(['Survived'], axis=1)
X_test = _input0.drop(['PassengerId'], axis=1)
(X.shape, y.shape, X_test.shape)
(X_train, X_valid, Y_train, Y_valid) = train_test_split(X, y, random_state=42, test_size=0.25)
(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier())