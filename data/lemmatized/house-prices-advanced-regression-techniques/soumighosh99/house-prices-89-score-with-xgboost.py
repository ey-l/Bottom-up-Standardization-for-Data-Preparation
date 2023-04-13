import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input1.columns
_input1 = _input1.drop(['Id'], axis=1)
_input1.info()
_input1.describe(include='all')
col_with_NA_as_category = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
null = _input1.isnull().sum()
null = null[null.values > 0]
null.sort_values(ascending=False)
null_col = [i for i in null.index if i not in col_with_NA_as_category]
null_col
_input1[['LotFrontage']] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(_input1[['LotFrontage']]))
_input1[['MasVnrArea']] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(_input1[['MasVnrArea']]))
_input1[['MasVnrType']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(_input1[['MasVnrType']]))
_input1[['Electrical']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(_input1[['Electrical']]))
_input1[['GarageYrBlt']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(_input1[['GarageYrBlt']]))
for col in col_with_NA_as_category:
    _input1[col] = _input1[col].fillna('Not Applicable', inplace=False)
_input1.isnull().sum()[_input1.isnull().sum().values > 0]
categorical_columns_with_numerical_values = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']
for i in categorical_columns_with_numerical_values:
    _input1[i] = _input1[i].astype(str)
corr = _input1.corr().sort_values(by='SalePrice', ascending=False)[['SalePrice']]
corr
col_to_be_removed = corr.index[7:]
_input1 = _input1.drop(col_to_be_removed, axis=1)
numeric_col = [i for i in corr.index[1:] if i not in col_to_be_removed]
categorical_col = [i for i in _input1.columns[:-1] if i not in numeric_col]
print('No. of Categorical columns: ', len(categorical_col))
print('No. of Numerical columns: ', len(numeric_col))
k = 1
plt.figure(figsize=(10, 8))
for col in numeric_col:
    plt.subplot(2, 3, k)
    sns.boxplot(x=_input1[col])
    k += 1

def remove_outlier(column):
    p25 = column.describe()[4]
    p75 = column.describe()[6]
    IQR = p75 - p25
    ul = p75 + 1.5 * IQR
    ll = p25 - 1.5 * IQR
    column = column.mask(column > ul, ul, inplace=False)
    column = column.mask(column < ll, ll, inplace=False)
for col in numeric_col:
    if col != 'YearRemodAdd':
        remove_outlier(_input1[col])
k = 1
plt.figure(figsize=(10, 8))
for col in numeric_col:
    plt.subplot(2, 3, k)
    sns.boxplot(x=_input1[col])
    k += 1
_input1[numeric_col] = StandardScaler().fit_transform(_input1[numeric_col])

def bar_plot(col):
    d = _input1[['SalePrice']].groupby(_input1[col]).mean().reset_index()
    sns.barplot(x=d[col], y=d['SalePrice'], order=d.sort_values(by='SalePrice', ascending=True)[col], palette='Greens')
    plt.title(col + 'vs. SalePrice')
plt.figure(figsize=(20, 20))
i = 1
for col in categorical_columns_with_numerical_values:
    plt.subplot(4, 4, i)
    bar_plot(col)
    i += 1
plt.figure(figsize=(20, 45))
i = 1
for col in [j for j in categorical_col if j not in categorical_columns_with_numerical_values]:
    plt.subplot(9, 5, i)
    bar_plot(col)
    i += 1
drop = ['MoSold', 'BsmtHalfBath', 'BsmtFullBath', 'YrSold', 'LandSlope']
_input1 = _input1.drop(drop, axis=1)
for i in drop:
    categorical_col.remove(i)
len(categorical_col)
for col in categorical_col:
    _input1[col] = LabelEncoder().fit_transform(_input1[col])
x = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=42, test_size=0.2)