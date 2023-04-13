import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
_input1.info()
_input1.head()
_input1 = _input1.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
_input0 = _input0.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
sex = pd.get_dummies(_input1['Sex'], drop_first=True)
embark = pd.get_dummies(_input1['Embarked'], drop_first=True)
_input1 = pd.concat([_input1, sex, embark], axis=1)
_input0 = pd.concat([_input0, sex, embark], axis=1)
_input1 = _input1.drop(['Sex', 'Embarked'], axis=1, inplace=False)
_input0 = _input0.drop(['Sex', 'Embarked'], axis=1, inplace=False)
imputer = SimpleImputer()
scaler = StandardScaler()
clf = LogisticRegression()
pipe = make_pipeline(imputer, scaler, clf)
features = _input1.drop('Survived', axis=1).columns
(X, y) = (_input1[features], _input1['Survived'])
_input0 = _input0.fillna(_input0.mean(), inplace=False)