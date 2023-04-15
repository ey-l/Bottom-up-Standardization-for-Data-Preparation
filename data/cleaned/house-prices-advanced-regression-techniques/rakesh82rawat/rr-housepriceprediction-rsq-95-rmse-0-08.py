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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train data')

print('\n', '---' * 20, '\n\n', 'Test data')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


print('TRAIN DATA INFO')
print(df_train.info(), '\n\n')
print('TEST DATA INFO')
print(df_test.info())
print('TRAIN DATA DESCRIPTION')
df_train.describe(include='all').T
print('TEST DATA DESCRIPTION')
df_test.describe(include='all').T
df_train.drop('Id', axis=1, inplace=True)
df_train['MSSubClass'] = df_train['MSSubClass'].astype('object')
df_train['YearBuilt'] = df_train['YearBuilt'].astype('object')
df_train['YearRemodAdd'] = df_train['YearRemodAdd'].astype('object')
df_train['MoSold'] = df_train['MoSold'].astype('object')
df_train['YrSold'] = df_train['YrSold'].astype('object')
df_train['MasVnrArea'] = df_train['MasVnrArea'].astype('float64')
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].astype('object')
df_test['MSSubClass'] = df_test['MSSubClass'].astype('object')
df_test['YearBuilt'] = df_train['YearBuilt'].astype('object')
df_test['YearRemodAdd'] = df_train['YearRemodAdd'].astype('object')
df_test['MoSold'] = df_train['MoSold'].astype('object')
df_test['YrSold'] = df_train['YrSold'].astype('object')
df_test['MasVnrArea'] = df_test['MasVnrArea'].astype('float64')
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].astype('object')
print('TRAIN Data - Missing values:', '\n', df_train.isna().sum()[df_train.isna().sum() > 0], '\n\n')

df_train['Alley'] = df_train['Alley'].fillna('NoAlley')
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('NoFireplace')
df_train['PoolQC'] = df_train['PoolQC'].fillna('NoPool')
df_train['Fence'] = df_train['Fence'].fillna('NoFence')
df_train['MiscFeature'] = df_train['MiscFeature'].fillna('NoMiscFeature')
df_train['GarageCond'] = df_train['GarageCond'].fillna('NoGarage')
df_train['GarageQual'] = df_train['GarageQual'].fillna('NoGarage')
df_train['GarageType'] = df_train['GarageType'].fillna('NoGarage')
df_train['GarageFinish'] = df_train['GarageFinish'].fillna('NoGarage')
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna('NoGarage')
df_train['BsmtFinSF2'] = df_train['BsmtFinSF2'].fillna('NoBasement')
df_train['BsmtFinSF1'] = df_train['BsmtFinSF1'].fillna('NoBasement')
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('NoBasement')
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna('NoBasement')
df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('NoBasement')
df_train['BsmtQual'] = df_train['BsmtQual'].fillna('NoBasement')
df_train['BsmtCond'] = df_train['BsmtCond'].fillna('NoBasement')
df_train['MasVnrType'] = df_train['MasVnrType'].fillna('NoMasVnr')
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(0)
df_test['Alley'] = df_test['Alley'].fillna('NoAlley')
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna('NoFireplace')
df_test['PoolQC'] = df_test['PoolQC'].fillna('NoPool')
df_test['Fence'] = df_test['Fence'].fillna('NoFence')
df_test['MiscFeature'] = df_test['MiscFeature'].fillna('NoMiscFeature')
df_test['GarageCond'] = df_test['GarageCond'].fillna('NoGarage')
df_test['GarageQual'] = df_test['GarageQual'].fillna('NoGarage')
df_test['GarageType'] = df_test['GarageType'].fillna('NoGarage')
df_test['GarageFinish'] = df_test['GarageFinish'].fillna('NoGarage')
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna('NoGarage')
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna('NoBasement')
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna('NoBasement')
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna('NoBasement')
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna('NoBasement')
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna('NoBasement')
df_test['BsmtQual'] = df_test['BsmtQual'].fillna('NoBasement')
df_test['BsmtCond'] = df_test['BsmtCond'].fillna('NoBasement')
df_test['MasVnrType'] = df_test['MasVnrType'].fillna('NoMasVnr')
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].median())
df_test[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']] = df_test[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']].fillna(df_test[['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']].median())
print('TRAIN Data - Missing values:', '\n', df_train.isna().sum()[df_train.isna().sum() > 0], '\n\n')
print('TEST Data - Missing values:', '\n', df_test.isna().sum()[df_test.isna().sum() > 0])
cols_mode = ['MSZoning', 'Electrical', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
for column in cols_mode:
    df_train[column].fillna(df_train[column].mode()[0], inplace=True)
    df_test[column].fillna(df_test[column].mode()[0], inplace=True)
print('TRAIN Data - Missing values:', '\n', df_train.isna().sum()[df_train.isna().sum() > 0], '\n\n')
print('TEST Data - Missing values:', '\n', df_test.isna().sum()[df_test.isna().sum() > 0])
(df_train.shape, df_test.shape)
(fig, ax) = plt.subplots()
ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4500) & (df_train['SalePrice'] < 300000)].index)

y_train = df_train['SalePrice']
df_train.drop('SalePrice', axis=1, inplace=True)
num_cols = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
cat_cols = [f for f in df_train.columns if df_train.dtypes[f] == 'object']
y_train_log = np.log(y_train)
df_train[num_cols].info()
plt.figure(figsize=(20, 15))
matrix = np.triu(df_train[num_cols].corr())
sns.heatmap(df_train[num_cols].corr(), annot=True, fmt='.2f', mask=matrix)

df_train.drop(df_train[['GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'PoolArea', 'PoolQC', 'MiscVal']], axis=1, inplace=True)
df_test.drop(df_test[['GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'PoolArea', 'PoolQC', 'MiscVal']], axis=1, inplace=True)
num_cols = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
cat_cols = [f for f in df_train.columns if df_train.dtypes[f] == 'object']
plt.figure(figsize=(20, 15))
matrix = np.triu(df_train[num_cols].corr())
sns.heatmap(df_train[num_cols].corr(), annot=True, fmt='.2f', mask=matrix)

plt.figure(figsize=(27, 25))
for i in range(len(num_cols)):
    plt.subplot(7, 5, i + 1)
    sns.histplot(data=df_train, x=df_train[num_cols[i]], kde=True)
    plt.title('Histplot of {}'.format(num_cols[i]))
    plt.tight_layout()
from scipy.stats import kurtosis, skew
skew(df_train[num_cols])
num_cols
num_cols_log = ['LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
df_train_log = df_train.copy(deep=True)

def log_transform(x):
    df_train_log[x] = np.log(df_train_log[x].values + 1)
for x in num_cols_log:
    log_transform(x)
df_test_log = df_test.copy(deep=True)

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