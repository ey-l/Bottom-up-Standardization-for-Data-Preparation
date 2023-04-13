import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
X_train = pd.get_dummies(_input1[features])
y_train = _input1['Survived']
final_X_test = pd.get_dummies(_input0[features])
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini', max_features='auto')
gs = GridSearchCV(cv=5, error_score=np.nan, estimator=classifier, param_grid={'min_samples_leaf': [10, 15, 20], 'max_depth': [3, 4, 5, 6], 'n_estimators': [10, 20, 30]})