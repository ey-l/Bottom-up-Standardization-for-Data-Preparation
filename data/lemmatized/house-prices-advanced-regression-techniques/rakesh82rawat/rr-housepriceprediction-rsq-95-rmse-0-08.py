import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import warnings
warnings.filterwarnings('ignore')
import sklearn
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train data')
print('\n', '---' * 20, '\n\n', 'Test data')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print('TRAIN DATA INFO')
print(_input1.info(), '\n\n')
print('TEST DATA INFO')
print(_input0.info())
print('TRAIN DATA DESCRIPTION')
_input1.describe(include='all').T
print('TEST DATA DESCRIPTION')
_input0.describe(include='all').T
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input1['MSSubClass'] = _input1['MSSubClass'].astype('object')
_input1['YearBuilt'] = _input1['YearBuilt'].astype('object')
_input1['YearRemodAdd'] = _input1['YearRemodAdd'].astype('object')
_input1['MoSold'] = _input1['MoSold'].astype('object')
_input1['YrSold'] = _input1['YrSold'].astype('object')
_input1['MasVnrArea'] = _input1['MasVnrArea'].astype('float64')
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].astype('object')
_input0['MSSubClass'] = _input0['MSSubClass'].astype('object')
_input0['YearBuilt'] = _input1['YearBuilt'].astype('object')
_input0['YearRemodAdd'] = _input1['YearRemodAdd'].astype('object')
_input0['MoSold'] = _input1['MoSold'].astype('object')
_input0['YrSold'] = _input1['YrSold'].astype('object')
_input0['MasVnrArea'] = _input0['MasVnrArea'].astype('float64')
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].astype('object')
print('TRAIN Data - Missing values:', '\n', _input1.isna().sum()[_input1.isna().sum() > 0], '\n\n')
_input1['Alley'] = _input1['Alley'].fillna('NoAlley')
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('NoFireplace')
_input1['PoolQC'] = _input1['PoolQC'].fillna('NoPool')
_input1['Fence'] = _input1['Fence'].fillna('NoFence')
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('NoMiscFeature')
_input1['GarageCond'] = _input1['GarageCond'].fillna('NoGarage')
_input1['GarageQual'] = _input1['GarageQual'].fillna('NoGarage')
_input1['GarageType'] = _input1['GarageType'].fillna('NoGarage')
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('NoGarage')
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna('NoGarage')
_input1['BsmtFinSF2'] = _input1['BsmtFinSF2'].fillna('NoBasement')
_input1['BsmtFinSF1'] = _input1['BsmtFinSF1'].fillna('NoBasement')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('NoBasement')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('NoBasement')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('NoBasement')
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('NoBasement')
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('NoBasement')
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('NoMasVnr')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input0['Alley'] = _input0['Alley'].fillna('NoAlley')
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('NoFireplace')
_input0['PoolQC'] = _input0['PoolQC'].fillna('NoPool')
_input0['Fence'] = _input0['Fence'].fillna('NoFence')
_input0['MiscFeature'] = _input0['MiscFeature'].fillna('NoMiscFeature')
_input0['GarageCond'] = _input0['GarageCond'].fillna('NoGarage')
_input0['GarageQual'] = _input0['GarageQual'].fillna('NoGarage')
_input0['GarageType'] = _input0['GarageType'].fillna('NoGarage')
_input0['GarageFinish'] = _input0['GarageFinish'].fillna('NoGarage')
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna('NoGarage')
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna('NoBasement')
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna('NoBasement')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna('NoBasement')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna('NoBasement')
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('NoBasement')
_input0['BsmtQual'] = _input0['BsmtQual'].fillna('NoBasement')
_input0['BsmtCond'] = _input0['BsmtCond'].fillna('NoBasement')
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('NoMasVnr')
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median())
_input0[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']] = _input0[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']].fillna(_input0[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']].median())
print('TRAIN Data - Missing values:', '\n', _input1.isna().sum()[_input1.isna().sum() > 0], '\n\n')
print('TEST Data - Missing values:', '\n', _input0.isna().sum()[_input0.isna().sum() > 0])
cols_mode = ['MSZoning', 'Electrical', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
for column in cols_mode:
    _input1[column] = _input1[column].fillna(_input1[column].mode()[0], inplace=False)
    _input0[column] = _input0[column].fillna(_input0[column].mode()[0], inplace=False)
print('TRAIN Data - Missing values:', '\n', _input1.isna().sum()[_input1.isna().sum() > 0], '\n\n')
print('TEST Data - Missing values:', '\n', _input0.isna().sum()[_input0.isna().sum() > 0])
(_input1.shape, _input0.shape)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4500) & (_input1['SalePrice'] < 300000)].index)
y_train = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1, inplace=False)
num_cols = [f for f in _input1.columns if _input1.dtypes[f] != 'object']
cat_cols = [f for f in _input1.columns if _input1.dtypes[f] == 'object']
y_train_log = np.log(y_train)
_input1[num_cols].info()
plt.figure(figsize=(20, 15))
matrix = np.triu(_input1[num_cols].corr())
sns.heatmap(_input1[num_cols].corr(), annot=True, fmt='.2f', mask=matrix)
_input1 = _input1.drop(_input1[['GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'PoolArea', 'PoolQC', 'MiscVal']], axis=1, inplace=False)
_input0 = _input0.drop(_input0[['GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'PoolArea', 'PoolQC', 'MiscVal']], axis=1, inplace=False)
num_cols = [f for f in _input1.columns if _input1.dtypes[f] != 'object']
cat_cols = [f for f in _input1.columns if _input1.dtypes[f] == 'object']
plt.figure(figsize=(20, 15))
matrix = np.triu(_input1[num_cols].corr())
sns.heatmap(_input1[num_cols].corr(), annot=True, fmt='.2f', mask=matrix)
plt.figure(figsize=(27, 25))
for i in range(len(num_cols)):
    plt.subplot(7, 5, i + 1)
    sns.histplot(data=_input1, x=_input1[num_cols[i]], kde=True)
    plt.title('Histplot of {}'.format(num_cols[i]))
    plt.tight_layout()
from scipy.stats import kurtosis, skew
skew(_input1[num_cols])
num_cols
num_cols_log = ['LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
df_train_log = _input1.copy(deep=True)

def log_transform(x):
    df_train_log[x] = np.log(df_train_log[x].values + 1)
for x in num_cols_log:
    log_transform(x)
df_test_log = _input0.copy(deep=True)

def log_transform(x):
    df_test_log[x] = np.log(df_test_log[x].values + 1)
for x in num_cols_log:
    log_transform(x)
plt.figure(figsize=(27, 25))
for i in range(len(num_cols)):
    plt.subplot(7, 5, i + 1)
    sns.histplot(data=df_train_log, x=df_train_log[num_cols[i]], kde=True)
    plt.title('Histplot of {}'.format(num_cols[i]))
    plt.tight_layout()
sns.histplot(y_train, kde=True)
plt.figure(figsize=(12, 7))
plt.subplot(221)
plot1 = sns.histplot(y_train, kde=True)
plot1.set_title('Before log transformation', fontsize=16)
plt.subplot(222)
plot2 = sns.histplot(y_train_log, kde=True)
plot2.set_title('After Log Transformation', fontsize=16)
plt.subplot(223)
sns.boxplot(y_train)
plt.subplot(224)
sns.boxplot(y_train_log)
df_train_log[num_cols].skew()
df_train_log_dummy = pd.get_dummies(df_train_log, columns=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'], drop_first=True)
cols_dummys = df_train_log_dummy.columns.tolist()
df_train_log_dummy[:3]
df_train_log_dummy.shape
df_test_log_dummy = pd.get_dummies(df_test_log, columns=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'], drop_first=True)
df_test_log_dummy = df_test_log_dummy.reindex(columns=cols_dummys).fillna(0)
df_test_log_dummy.head()
df_test_log_dummy.shape
X_train_log_dummy = df_train_log_dummy.copy(deep=True)
from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()