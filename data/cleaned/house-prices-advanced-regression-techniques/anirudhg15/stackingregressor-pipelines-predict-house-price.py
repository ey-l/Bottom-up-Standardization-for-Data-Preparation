import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data
print(train_data.apply(lambda col: col.dtypes))
categorical_list = list(test_data.select_dtypes(include='object'))
train_data['train'] = 1
test_data['train'] = 0
combined = pd.concat([train_data, test_data])
for col in categorical_list:
    combined[col] = combined[col].astype('category').cat.codes
train_data = combined[combined['train'] == 1]
test_data = combined[combined['train'] == 0]
train_data = train_data.drop(['train', 'Id'], axis=1)
ids = test_data['Id']
test_data = test_data.drop(['Id', 'train', 'SalePrice'], axis=1)
numeric_transformer = Pipeline(steps=[('imputer', IterativeImputer(random_state=42)), ('robust', RobustScaler(quantile_range=(10.0, 90.0))), ('minmax', MinMaxScaler())])
categorical_transformer = Pipeline(steps=[('imputer', IterativeImputer(random_state=42))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, selector(dtype_exclude='object')), ('cat', categorical_transformer, selector(dtype_include='object'))])
estimators = [('ridge', KernelRidge(alpha=14.1)), ('lasso', Lasso(max_iter=2500, alpha=0.0008, random_state=42)), ('svr', SVR(C=3.0, max_iter=2500, epsilon=0.008, gamma=0.0003)), ('xgb', XGBRegressor(learning_rate=0.01, max_depth=3, gamma=0, sub_sample=0.7, seed=42))]
clf = Pipeline(steps=[('preprocessor', preprocessor), ('rfecv', RFECV(RandomForestRegressor(random_state=42), step=1, cv=2)), ('model', StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)))])
y_train = train_data['SalePrice']
x_train = train_data.drop('SalePrice', axis=1)
(x_train, x_valid, y_train, y_valid) = train_test_split(x_train, y_train, test_size=0.2, random_state=42)