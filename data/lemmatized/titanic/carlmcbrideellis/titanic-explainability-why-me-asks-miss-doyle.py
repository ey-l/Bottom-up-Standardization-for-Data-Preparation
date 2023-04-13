import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
X_train = pd.get_dummies(_input1[features])
y_train = _input1['Survived']
X_test = pd.get_dummies(_input0[features])
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', fit_intercept=False)