import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics as metrics
import math
from scipy.stats import norm, skew
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.head()
print(_input1['SalePrice'].describe())
sns.distplot(_input1['SalePrice'], color='red')
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], color='red', fit=norm)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
corr = _input1.corr()
highest_corr_features = corr.index[abs(corr['SalePrice']) > 0.5]
plt.figure(figsize=(10, 10))
g = sns.heatmap(_input1[highest_corr_features].corr(), annot=True, cmap='RdYlGn')
corr['SalePrice'].sort_values(ascending=False)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols])
y_train = _input1['SalePrice']
test_id = _input0['Id']
all_data = pd.concat([_input1, _input0], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)
Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
all_data = all_data.drop(missing_data[missing_data['Total'] > 5].index, axis=1, inplace=False)
print(all_data.isnull().sum().max())
total = all_data.isnull().sum().sort_values(ascending=False)
total.head(19)
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']
for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)
categorical_missed = ['Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Electrical', 'KitchenQual']
for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data = all_data.drop(['Utilities'], axis=1, inplace=False)
all_data.isnull().sum().max()
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew
for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data = pd.get_dummies(all_data)
all_data.head()
x_train = all_data[:len(y_train)]
x_test = all_data[len(y_train):]
(x_test.shape, x_train.shape)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)