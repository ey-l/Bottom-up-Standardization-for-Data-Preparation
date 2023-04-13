import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
plt.style.use(style='fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 6)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print('No. of records in train dataset: ', len(_input1.index))
print('No. of columns in train dataset: ', len(_input1.columns))
print('No. of records in test dataset: ', len(_input0.index))
print('No. of columns in test dataset: ', len(_input0.columns))
print('Total missing values in train set', sum(_input1.isna().sum()))
print('Total missing values in test set', sum(_input0.isna().sum()))
print('Total missing values in train set', sum(_input1.isna().sum()))
print('Total missing values in test set', sum(_input0.isna().sum()))
_input1['SalePrice'].describe()
numeric_cols = _input1.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
(print('The Most Correlated Features with SalePrice:'), print(corr['SalePrice'].sort_values(ascending=False)[:10], '\n'))
(print('The Most Uncorrelated Features with SalePrice:'), print(corr['SalePrice'].sort_values(ascending=False)[-5:]))
plt.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea (Above grade "ground" living area square feet)')
plt.scatter(x=_input1['GarageArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
_input1 = _input1[_input1['GrLivArea'] < 4500]
_input1 = _input1[_input1['GarageArea'] < 1200]
train_percentage = _input1.isnull().sum() / _input1.shape[0]
print(train_percentage[train_percentage > 0.8])
_input1 = _input1.drop(train_percentage[train_percentage > 0.8].index, axis=1)
test_percentage = _input0.isnull().sum() / _input0.shape[0]
print(test_percentage[test_percentage > 0.8])
_input0 = _input0.drop(test_percentage[test_percentage > 0.8].index, axis=1)
le = preprocessing.LabelEncoder()
for name in _input1.columns:
    if _input1[name].dtypes == 'O':
        _input1[name] = _input1[name].astype(str)