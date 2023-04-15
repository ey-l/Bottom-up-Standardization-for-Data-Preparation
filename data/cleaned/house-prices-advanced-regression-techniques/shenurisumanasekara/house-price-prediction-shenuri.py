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
train_set = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_set = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(len(train_set))
print(len(test_set))
train_set.head()
train_set.info()
print(len(train_set.columns))
print(train_set.isnull().sum().sort_values(ascending=False))
train_set = train_set.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
train_set.describe()
print(len(train_set.columns))
categorical_data = train_set.select_dtypes(['object']).columns
train_set[categorical_data] = train_set[categorical_data].fillna(train_set[categorical_data].mode().iloc[0])
train_set[categorical_data].mode()
numerical_data = train_set.select_dtypes(['float64', 'int64']).columns
train_set[numerical_data] = train_set[numerical_data].fillna(train_set[numerical_data].mean())
train_set[numerical_data].mean()
print(train_set.isnull().sum().sort_values(ascending=False))
train_set.hist(figsize=(20, 20), bins=20)

train_set = train_set.drop(columns=['LowQualFinSF', 'PoolArea', 'MiscVal', '3SsnPorch'])
print(len(train_set.columns))
print(train_set.dtypes)
category_columns = train_set.select_dtypes(['object']).columns
print(category_columns)
train_set[category_columns] = train_set[category_columns].astype('category').apply(lambda x: x.cat.codes)
float_columns = train_set.select_dtypes(['float64']).columns
print(float_columns)
train_set['LotFrontage'] = pd.to_numeric(train_set['LotFrontage'], errors='coerce')
train_set['MasVnrArea'] = pd.to_numeric(train_set['MasVnrArea'], errors='coerce')
train_set['GarageYrBlt'] = pd.to_numeric(train_set['GarageYrBlt'], errors='coerce')
train_set['SalePrice'] = pd.to_numeric(train_set['SalePrice'], errors='coerce')
train_set = train_set.astype('int64')
print(train_set.dtypes)
train_set['SalePrice'].describe()
sns.displot(train_set['SalePrice'])
correlation_matrix = train_set.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
correlation_num = 30
correlation_cols = correlation_matrix.nlargest(correlation_num, 'SalePrice')['SalePrice'].index
correlation_mat_sales = np.corrcoef(train_set[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)

y = train_set['SalePrice']
x = train_set.drop(columns=['SalePrice', 'Id'])
print(len(x.columns))
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
linear_model = LinearRegression()