import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.shape
_input0.shape
_input1.info()
_input1.isnull().sum()
lst = _input1.columns
nan_cols = []
for i in lst:
    if _input1[i].isnull().sum() > 0:
        nan_cols.append(i)
print(nan_cols)
lst1 = _input0.columns
nan_cols1 = []
for i in lst1:
    if _input0[i].isnull().sum() > 0:
        nan_cols1.append(i)
print(nan_cols1)
nan_train_num_cols = [col for col in nan_cols if _input1[col].dtypes != 'O']
nan_train_cat_cols = [col for col in nan_cols if _input1[col].dtypes == 'O']
nan_test_num_cols1 = [col for col in nan_cols1 if _input0[col].dtypes != 'O']
nan_test_cat_cols1 = [col for col in nan_cols1 if _input0[col].dtypes == 'O']
for j in nan_train_num_cols:
    _input1[j] = _input1[j].fillna(_input1[j].mean(), inplace=False)
for k in nan_train_cat_cols:
    _input1[k] = _input1[k].fillna(_input1[k].mode()[0], inplace=False)
for j in nan_test_num_cols1:
    _input0[j] = _input0[j].fillna(_input0[j].mean(), inplace=False)
for k in nan_test_cat_cols1:
    _input0[k] = _input0[k].fillna(_input0[k].mode()[0], inplace=False)
_input1.head()
cat_cols = [cols for cols in _input1.columns if _input1[cols].dtypes == 'O']
cat_cols1 = [cols for cols in _input0.columns if _input0[cols].dtypes == 'O']
cat_cols == cat_cols1
X = _input1.drop('SalePrice', axis=1)
Y = _input1['SalePrice']
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
ct = ColumnTransformer([('step1', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)])
pipeline = Pipeline([('cltf_step', ct), ('Gradient Boost', XGBRegressor(learning_rate=1, random_state=42, n_jobs=5))])