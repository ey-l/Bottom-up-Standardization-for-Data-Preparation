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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print('No. of records in train dataset: ', len(train.index))
print('No. of columns in train dataset: ', len(train.columns))
print('No. of records in test dataset: ', len(test.index))
print('No. of columns in test dataset: ', len(test.columns))
print('Total missing values in train set', sum(train.isna().sum()))
print('Total missing values in test set', sum(test.isna().sum()))
print('Total missing values in train set', sum(train.isna().sum()))
print('Total missing values in test set', sum(test.isna().sum()))
train['SalePrice'].describe()
numeric_cols = train.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
(print('The Most Correlated Features with SalePrice:'), print(corr['SalePrice'].sort_values(ascending=False)[:10], '\n'))
(print('The Most Uncorrelated Features with SalePrice:'), print(corr['SalePrice'].sort_values(ascending=False)[-5:]))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea (Above grade "ground" living area square feet)')
plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
train = train[train['GrLivArea'] < 4500]
train = train[train['GarageArea'] < 1200]
train_percentage = train.isnull().sum() / train.shape[0]
print(train_percentage[train_percentage > 0.8])
train = train.drop(train_percentage[train_percentage > 0.8].index, axis=1)
test_percentage = test.isnull().sum() / test.shape[0]
print(test_percentage[test_percentage > 0.8])
test = test.drop(test_percentage[test_percentage > 0.8].index, axis=1)
le = preprocessing.LabelEncoder()
for name in train.columns:
    if train[name].dtypes == 'O':
        train[name] = train[name].astype(str)