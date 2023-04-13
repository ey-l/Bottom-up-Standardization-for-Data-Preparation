import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1.head()
y = _input1.Transported
X = _input1.drop(['Transported'], axis=1)
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and X_train[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_t = X_train[my_cols].copy()
X_v = X_val[my_cols].copy()
X_test = _input0[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder())])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
from sklearn.metrics import mean_absolute_error
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])