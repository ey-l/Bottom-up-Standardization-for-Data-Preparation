import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1 = _input1.set_index('PassengerId', drop=True)
_input1.head()
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input0 = _input0.set_index('PassengerId', drop=True)
_input0.head()
_input1.Ticket = pd.to_numeric(_input1.Ticket.str.split().str[-1], errors='coerce')
_input1.info()
_input0.Ticket = pd.to_numeric(_input0.Ticket.str.split().str[-1], errors='coerce')
_input0.info()
_input1 = _input1.drop(['Name', 'Cabin'], axis=1)
_input0 = _input0.drop(['Name', 'Cabin'], axis=1)
_input1.head()
_input0.head()
_input1 = pd.get_dummies(_input1, columns=['Sex', 'Embarked'])
_input0 = pd.get_dummies(_input0, columns=['Sex', 'Embarked'])
_input1.head()
_input0.head()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input1['Ticket'] = _input1['Ticket'].fillna(_input1['Ticket'].median())
_input1.info()
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median())
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].median())
_input0.info()
y = _input1['Survived']
_input1 = _input1.drop(['Survived', 'Sex_male', 'Embarked_S'], axis=1)
X = _input1
_input0 = _input0.drop(['Sex_male', 'Embarked_S'], axis=1)
XTEST = _input0
XTEST.head()
X.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
scores = cross_val_score(clf, X_scaled, y, cv=5).mean()
scores
from xgboost import XGBClassifier
clf = XGBClassifier(colsample_bylevel=0.9, colsample_bytree=0.8, gamma=0.99, max_depth=5, min_child_weight=1, n_estimators=10, nthread=4, random_state=2, silent=True)
scores = cross_val_score(clf, X_scaled, y, cv=5).mean()
scores