import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data.head()
test_data.head()
data.info()
data.describe()
data.isna().sum()
object_columns = []
for i in data.columns:
    if data[i].dtype == object:
        object_columns.append(i)
num_columns = list(set(list(data.columns)) - set(object_columns))
num_columns
object_data = data[object_columns]
num_data = data[num_columns]
num_data.info()
object_data.info()
object_data['SalePrice'] = data['SalePrice']
for i in object_columns:
    object_data[i].fillna('No', inplace=True)
object_data['PoolQC'].unique()
for i in object_columns:
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.barplot(x='SalePrice', y=i, data=object_data)
    plt.title('{} / SalePrice'.format(i))

for i in object_columns:
    data[i] = object_data[i]
data.head()
test_data
t_object_columns = []
for i in test_data.columns:
    if test_data[i].dtype == object:
        t_object_columns.append(i)
t_object_data = test_data[t_object_columns]
for i in t_object_columns:
    t_object_data[i].fillna('No', inplace=True)
for i in t_object_columns:
    test_data[i] = t_object_data[i]
data[object_columns]
test_data[t_object_columns]
t_num_columns = list(set(list(test_data.columns)) - set(t_object_columns))
len(set(t_num_columns) - set(num_columns))
set(num_columns) - set(t_num_columns)
data = pd.get_dummies(data, columns=object_columns, drop_first=True)
data.shape
test_data = pd.get_dummies(test_data, columns=t_object_columns, drop_first=True)
test_data.shape
data_unique_columns = list(set(data.columns) - set(test_data.columns))
data_unique_columns
test_unique_columns = list(set(test_data.columns) - set(data.columns))
test_unique_columns
zeros = np.zeros(len(data['Id']))
t_zeros = np.zeros(len(test_data['Id']))
(zeros.shape, t_zeros.shape)
for i in test_unique_columns:
    data[i] = zeros
for i in data_unique_columns:
    test_data[i] = t_zeros
(data.shape, test_data.shape)
x = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']
import xgboost as xgb
pre_model = xgb.XGBRegressor()
parameters = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [5, 7, 10], 'subsample': [0.5, 0.7, 1], 'n_estimators': [300, 500, 1000]}
from sklearn.model_selection import RandomizedSearchCV
model_rs = RandomizedSearchCV(pre_model, param_distributions=parameters, n_iter=30)