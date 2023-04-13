import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_labels = _input1.pop('Transported')
_input1
_input1 = _input1.drop(['Name', 'Cabin'], axis=1, inplace=False)
categorical_cols = _input1.select_dtypes(['bool_', 'object_']).columns
numeric_cols = _input1.select_dtypes(exclude=['bool_', 'object_']).columns
categorical_cols
categorical_cols = categorical_cols.drop('PassengerId')
numeric_cols
_input1
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
_input1[categorical_cols] = encoder.fit_transform(_input1[categorical_cols])
_input1[categorical_cols]
_input1.isna().sum()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iterative_imputer = IterativeImputer()
_input1[numeric_cols] = pd.DataFrame(iterative_imputer.fit_transform(_input1[numeric_cols]), columns=numeric_cols)
from sklearn.impute import SimpleImputer
categorical_imputer = SimpleImputer(strategy='most_frequent')
_input1[categorical_cols] = pd.DataFrame(categorical_imputer.fit_transform(_input1[categorical_cols]), columns=categorical_cols)
_input1.isna().sum()
_input1['group'] = _input1['PassengerId'].str.split('_').str[0]
_input1['group'] = pd.to_numeric(_input1['group'])
_input1['group']
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_col_names = [col + '_scaled' for col in numeric_cols]
_input1[new_col_names] = scaler.fit_transform(_input1[numeric_cols])
_input1[new_col_names]
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(_input1, train_labels)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=_input1.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores
_input1 = _input1.drop(['Destination', 'VIP'], axis=1, inplace=False)
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(_input1, train_labels, train_size=0.8)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)