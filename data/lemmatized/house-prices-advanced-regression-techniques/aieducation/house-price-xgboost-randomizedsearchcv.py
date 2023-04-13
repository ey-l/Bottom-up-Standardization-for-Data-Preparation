import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input1.describe()
_input1.isna().sum()
object_columns = []
for i in _input1.columns:
    if _input1[i].dtype == object:
        object_columns.append(i)
num_columns = list(set(list(_input1.columns)) - set(object_columns))
num_columns
object_data = _input1[object_columns]
num_data = _input1[num_columns]
num_data.info()
object_data.info()
object_data['SalePrice'] = _input1['SalePrice']
for i in object_columns:
    object_data[i] = object_data[i].fillna('No', inplace=False)
object_data['PoolQC'].unique()
for i in object_columns:
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.barplot(x='SalePrice', y=i, data=object_data)
    plt.title('{} / SalePrice'.format(i))
for i in object_columns:
    _input1[i] = object_data[i]
_input1.head()
_input0
t_object_columns = []
for i in _input0.columns:
    if _input0[i].dtype == object:
        t_object_columns.append(i)
t_object_data = _input0[t_object_columns]
for i in t_object_columns:
    t_object_data[i] = t_object_data[i].fillna('No', inplace=False)
for i in t_object_columns:
    _input0[i] = t_object_data[i]
_input1[object_columns]
_input0[t_object_columns]
t_num_columns = list(set(list(_input0.columns)) - set(t_object_columns))
len(set(t_num_columns) - set(num_columns))
set(num_columns) - set(t_num_columns)
_input1 = pd.get_dummies(_input1, columns=object_columns, drop_first=True)
_input1.shape
_input0 = pd.get_dummies(_input0, columns=t_object_columns, drop_first=True)
_input0.shape
data_unique_columns = list(set(_input1.columns) - set(_input0.columns))
data_unique_columns
test_unique_columns = list(set(_input0.columns) - set(_input1.columns))
test_unique_columns
zeros = np.zeros(len(_input1['Id']))
t_zeros = np.zeros(len(_input0['Id']))
(zeros.shape, t_zeros.shape)
for i in test_unique_columns:
    _input1[i] = zeros
for i in data_unique_columns:
    _input0[i] = t_zeros
(_input1.shape, _input0.shape)
x = _input1.drop(['Id', 'SalePrice'], axis=1)
y = _input1['SalePrice']
import xgboost as xgb
pre_model = xgb.XGBRegressor()
parameters = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [5, 7, 10], 'subsample': [0.5, 0.7, 1], 'n_estimators': [300, 500, 1000]}
from sklearn.model_selection import RandomizedSearchCV
model_rs = RandomizedSearchCV(pre_model, param_distributions=parameters, n_iter=30)