import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
_input1 = pd.read_csv('data/input/titanic/train.csv', index_col='PassengerId')
_input1.isnull().sum() * 100 / _input1.shape[0]
_input1 = _input1.drop(['Name', 'Ticket', 'Cabin'], axis=1)
X = _input1.drop('Survived', axis=1)
y = _input1['Survived']
numerical = [col for col in X.select_dtypes(exclude='object')]
categorial = [col for col in X.select_dtypes(include='object')]
numerical_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='constant'))])
categorial_pipeline = Pipeline(steps=[('impute_cat', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))])
preprocessing = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical), ('cat', categorial_pipeline, categorial)])
model_2 = GradientBoostingClassifier()
pipeline_2 = Pipeline(steps=[('preprocessing', preprocessing), ('model', model_2)])
param = {'model__n_estimators': np.arange(50, 1000, 100)}
grid = GridSearchCV(pipeline_2, param_grid=param, cv=5)