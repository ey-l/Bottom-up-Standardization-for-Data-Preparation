import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
y = _input1['Transported']
X = _input1.drop('Transported', axis=1)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
cat_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == 'object']
num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
all_cols = cat_cols + num_cols
X_train = X_train_full[all_cols].copy()
X_valid = X_valid_full[all_cols].copy()
X_test = _input0[all_cols].copy()
print(f'CATEGORY: {cat_cols}\n')
print(f'NUMERIC: {num_cols}\n')
num_transform = SimpleImputer(strategy='constant')
cat_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', num_transform, num_cols), ('cat', cat_transform, cat_cols)])
model = RandomForestRegressor(n_estimators=100, random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])