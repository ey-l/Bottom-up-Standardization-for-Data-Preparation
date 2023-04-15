import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
X_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
best_alpha = 10
my_model_1 = Lasso(alpha=best_alpha, max_iter=50000)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', my_model_1)])