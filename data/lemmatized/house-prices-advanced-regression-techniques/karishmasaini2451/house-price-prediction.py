import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_ids = _input0['Id'].values
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.head()
_input1.shape
_input1.columns
_input1.info()
_input1.dtypes
_input1.isna().sum()
_input1['MSZoning'] = _input1['MSZoning'].replace('C (all)', 'C')
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median(), inplace=False)
obj_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
for i in obj_NA:
    _input1[i] = _input1[i].fillna('NA', inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('CBlock', inplace=False)
drop_na = ['MasVnrArea', 'Electrical']
for i in drop_na:
    print(i, ':', round(_input1[i].isna().sum() / _input1.shape[0] * 100, 2))
_input1 = _input1.dropna(subset=drop_na, axis=0)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mode()[0], inplace=False)
round(_input1['MiscFeature'].isna().sum() / _input1.shape[0] * 100, 2)
_input1 = _input1.drop(columns=['MiscFeature'], inplace=False)
_input0['MSZoning'] = _input0['MSZoning'].replace('C (all)', 'C')
obj_mode = ['MSZoning', 'MasVnrType', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'SaleType']
for i in obj_mode:
    _input0[i] = _input0[i].fillna(_input0[i].mode()[0], inplace=False)
obj_median = ['LotFrontage', 'MasVnrArea']
for i in obj_median:
    _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
obj_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
for i in obj_NA:
    _input0[i] = _input0[i].fillna('NA', inplace=False)
_input0['Utilities'] = _input0['Utilities'].fillna('NoSeWa', inplace=False)
_input0['Exterior1st'] = _input0['Exterior1st'].fillna('Stone', inplace=False)
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna('Other', inplace=False)
round(_input0['MiscFeature'].isna().sum() / _input0.shape[0] * 100, 2)
_input0 = _input0.drop(columns=['MiscFeature'], inplace=False)
int_list = ['LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
for i in int_list:
    _input1[i] = _input1[i].astype('int64')
    _input0[i] = _input0[i].astype('int64')
_input1.SalePrice.hist(bins=50, figsize=(15, 5))
plt.title('Highly preffered price range?')
plt.figure(figsize=(20, 20))
plt.subplot(4, 3, 1)
plt.title('LotArea')
plt.scatter(x='LotArea', y='SalePrice', data=_input1)
plt.subplot(4, 3, 2)
plt.title('GrLivArea')
plt.scatter(x='GrLivArea', y='SalePrice', data=_input1)
plt.subplot(4, 3, 3)
plt.title('GarageArea')
plt.scatter(x='GarageArea', y='SalePrice', data=_input1)
plt.subplot(4, 3, 4)
plt.title('PoolArea')
plt.scatter(x='PoolArea', y='SalePrice', data=_input1)
plt.subplot(4, 3, 5)
plt.title('TotalBsmtSF')
plt.scatter(x='TotalBsmtSF', y='SalePrice', data=_input1)
plt.subplot(4, 3, 6)
plt.title('WoodDeckSF')
plt.scatter(x='WoodDeckSF', y='SalePrice', data=_input1)
plt.subplot(4, 3, 7)
plt.title('OpenPorchSF')
plt.scatter(x='OpenPorchSF', y='SalePrice', data=_input1)
plt.subplot(4, 3, 8)
plt.title('EnclosedPorch')
plt.scatter(x='EnclosedPorch', y='SalePrice', data=_input1)
plt.subplot(4, 3, 9)
plt.title('3SsnPorch')
plt.scatter(x='3SsnPorch', y='SalePrice', data=_input1)
plt.subplot(4, 3, 10)
plt.title('1stFlrSF')
plt.scatter(x='1stFlrSF', y='SalePrice', data=_input1)
plt.subplot(4, 3, 11)
plt.title('2ndFlrSF')
plt.scatter(x='2ndFlrSF', y='SalePrice', data=_input1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='HouseStyle', y='SalePrice', data=_input1)
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='RoofStyle', y='SalePrice', data=_input1)
plt.xticks(rotation=45)
plt.figure(figsize=(15, 5))
sns.barplot(x='Neighborhood', y='SalePrice', data=_input1)
plt.xticks(rotation=90)
plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1)
sns.barplot(x='BsmtFullBath', y='SalePrice', data=_input1)
plt.subplot(3, 3, 2)
sns.barplot(x='BsmtHalfBath', y='SalePrice', data=_input1)
plt.subplot(3, 3, 3)
sns.barplot(x='FullBath', y='SalePrice', data=_input1)
plt.subplot(3, 3, 4)
sns.barplot(x='HalfBath', y='SalePrice', data=_input1)
plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1)
sns.barplot(x='OverallQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 2)
sns.barplot(x='OverallCond', y='SalePrice', data=_input1)
plt.subplot(3, 3, 3)
sns.barplot(x='ExterQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 4)
sns.barplot(x='ExterCond', y='SalePrice', data=_input1)
plt.subplot(3, 3, 5)
sns.barplot(x='BsmtQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 6)
sns.barplot(x='BsmtCond', y='SalePrice', data=_input1)
plt.subplot(3, 3, 7)
sns.barplot(x='KitchenQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 8)
sns.barplot(x='GarageQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 9)
sns.barplot(x='GarageCond', y='SalePrice', data=_input1)
yearsold = ['2006', '2007', '2008', '2009', '2010']
values = _input1['YrSold'].value_counts()
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.pie(values, labels=yearsold, explode=(0.1, 0.1, 0.1, 0.1, 0.1), autopct='%.2f%%', startangle=45)
plt.title('')
monthsold = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
values = _input1['MoSold'].value_counts()
plt.subplot(1, 2, 2)
plt.pie(values, labels=monthsold, explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), autopct='%.2f%%', startangle=45)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in _input1.columns:
    _input1[i] = label_encoder.fit_transform(_input1[i])
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in _input0.columns:
    _input0[i] = label_encoder.fit_transform(_input0[i])
X = _input1.drop(columns=['SalePrice'])
y = _input1['SalePrice']
X_scaled = StandardScaler()
X_scaled = X_scaled.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model1 = AdaBoostRegressor()