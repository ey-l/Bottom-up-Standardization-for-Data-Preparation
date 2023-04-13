import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1 = _input1.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'Embarked', 'PassengerId'], axis=1, inplace=False)
_input1.loc[_input1['Sex'] == 'male', 'Sex'] = 1
_input1.loc[_input1['Sex'] == 'female', 'Sex'] = 0
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
train_features = _input1.drop('Survived', axis=1)
train_target = _input1['Survived']
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
DT = DecisionTreeClassifier()
LR = LogisticRegression()
RF = RandomForestClassifier(n_estimators=1000)
model = [DT, LR, RF]
parameters_dt = {'max_depth': [1, 3, 5, 10], 'min_samples_leaf': [1, 3, 5, 10]}
grid_cv_dt = GridSearchCV(model[0], param_grid=parameters_dt, scoring='accuracy', cv=5)