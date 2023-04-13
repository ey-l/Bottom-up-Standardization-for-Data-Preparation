import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
pd.set_option('display.max_columns', None)
import category_encoders as ce
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from pandas_profiling import ProfileReport
import math

def save_file(predictions):
    """Save submission file."""
    output = pd.DataFrame({'Id': sample_submission_file.Id, 'SalePrice': predictions})
    print('Submission file is saved')

def calculate_root_mean_squared_log_error(y_true, y_pred):
    """Calculate root mean squared error of log(y_true) and log(y_pred)"""
    if len(y_pred) != len(y_true):
        return 'error_mismatch'
    y_pred_new = [math.log(x + 1) for x in y_pred]
    y_true_new = [math.log(x + 1) for x in y_true]
    return mean_squared_error(y_true_new, y_pred_new, squared=False)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X = _input1.copy()
X = X.dropna(axis=0, subset=['SalePrice'], inplace=False)
y = X.SalePrice
X = X.drop(['SalePrice', 'Utilities'], axis=1, inplace=False)
_input0 = _input0.drop(['Utilities'], axis=1, inplace=False)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print('Data is OK')
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
missing_val_count_by_column_train = X_train.isnull().sum()
print('Number of missing values in each column:')
print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
missing_val_count_by_column_numeric = X_train[numerical_cols].isnull().sum()
print('Number of missing values in numerical columns:')
print(missing_val_count_by_column_numeric[missing_val_count_by_column_numeric > 0])
constant_num_cols = ['GarageYrBlt', 'MasVnrArea']
mean_num_cols = list(set(numerical_cols).difference(set(constant_num_cols)))
constant_categorical_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
mf_categorical_cols = list(set(categorical_cols).difference(set(constant_categorical_cols)))
my_cols = constant_num_cols + mean_num_cols + constant_categorical_cols + mf_categorical_cols
numerical_transformer_m = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
numerical_transformer_c = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])
categorical_transformer_mf = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
categorical_transformer_c = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='NA')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer(transformers=[('num_mean', numerical_transformer_m, mean_num_cols), ('num_constant', numerical_transformer_c, constant_num_cols), ('cat_mf', categorical_transformer_mf, mf_categorical_cols), ('cat_c', categorical_transformer_c, constant_categorical_cols)])
model = XGBRegressor(learning_rate=0.01, n_estimators=2500, max_depth=4, min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.25)