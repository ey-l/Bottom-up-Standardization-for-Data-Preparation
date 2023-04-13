import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input1.head()
_input1.describe()
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1.Age.mean(), inplace=False)
_input1['Embarked'] = _input1['Embarked'].fillna('S', inplace=False)
_input1.isnull().sum()
x_train = _input1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp']]
x_train = pd.get_dummies(x_train)
x_train.head()
y_train = _input1[['Survived']]
y_train.head()
clf = RandomForestClassifier(random_state=10, max_features='sqrt')
pipe = Pipeline([('classify', clf)])
param = {'classify__n_estimators': list(range(20, 30, 1)), 'classify__max_depth': list(range(3, 10, 1))}
grid = GridSearchCV(estimator=pipe, param_grid=param, scoring='accuracy', cv=10)