import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 100
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(len(_input1))
print(len(_input0))
_input1.head()
_input1.info()
print(len(_input1.columns))
print(_input1.isnull().sum().sort_values(ascending=False))
_input1 = _input1.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
_input1.describe()
print(len(_input1.columns))
categorical_data = _input1.select_dtypes(['object']).columns
_input1[categorical_data] = _input1[categorical_data].fillna(_input1[categorical_data].mode().iloc[0])
_input1[categorical_data].mode()
numerical_data = _input1.select_dtypes(['float64', 'int64']).columns
_input1[numerical_data] = _input1[numerical_data].fillna(_input1[numerical_data].mean())
_input1[numerical_data].mean()
print(_input1.isnull().sum().sort_values(ascending=False))
_input1.hist(figsize=(20, 20), bins=20)
_input1 = _input1.drop(columns=['LowQualFinSF', 'PoolArea', 'MiscVal', '3SsnPorch'])
print(len(_input1.columns))
print(_input1.dtypes)
category_columns = _input1.select_dtypes(['object']).columns
print(category_columns)
_input1[category_columns] = _input1[category_columns].astype('category').apply(lambda x: x.cat.codes)
float_columns = _input1.select_dtypes(['float64']).columns
print(float_columns)
_input1['LotFrontage'] = pd.to_numeric(_input1['LotFrontage'], errors='coerce')
_input1['MasVnrArea'] = pd.to_numeric(_input1['MasVnrArea'], errors='coerce')
_input1['GarageYrBlt'] = pd.to_numeric(_input1['GarageYrBlt'], errors='coerce')
_input1['SalePrice'] = pd.to_numeric(_input1['SalePrice'], errors='coerce')
_input1 = _input1.astype('int64')
print(_input1.dtypes)
_input1['SalePrice'].describe()
sns.displot(_input1['SalePrice'])
correlation_matrix = _input1.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
correlation_num = 30
correlation_cols = correlation_matrix.nlargest(correlation_num, 'SalePrice')['SalePrice'].index
correlation_mat_sales = np.corrcoef(_input1[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)
y = _input1['SalePrice']
x = _input1.drop(columns=['SalePrice', 'Id'])
print(len(x.columns))
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
linear_model = LinearRegression()