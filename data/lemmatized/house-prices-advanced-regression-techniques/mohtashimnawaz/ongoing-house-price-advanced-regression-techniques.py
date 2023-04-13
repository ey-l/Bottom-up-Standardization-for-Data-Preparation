import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Shape of trainig set:', _input1.shape)
print('Shape of test set:', _input0.shape)
y = _input1['SalePrice']
total = _input1.append(_input0)
print(total.shape)
mask = total.isnull().sum() > 0
print(sum(mask))
total.isnull().sum()
total.columns
print('mean of SalePrice', np.mean(y))
print('Median of Sale Price', np.median(y))
print('Minimum Sale Price', np.min(y))
print('Maximum Sale Price', np.max(y))
plt.figure(figsize=(10, 8))
sns.distplot(y, label='Distribution of Sale Price')
plt.figure(figsize=(10, 8))
sns.boxplot(data=y)
plt.xlabel('Outlier detection from Sale Price')
print('Number of possible outlier points above 600000=', sum(y > 600000))
print('Number of possible outlier points above 700000=', sum(y > 700000))
plt.figure(figsize=(10, 8))
sns.countplot(x='MSSubClass', data=_input1[_input1['SalePrice'] < 700000])
plt.figure(figsize=(8, 6))
sns.countplot(x='MSZoning', data=_input1[_input1['SalePrice'] < 700000])
_input1['SalePrice'].groupby(_input1.MSZoning).median()
plt.figure(figsize=(8, 6))
sns.countplot(x='SaleType', data=_input1[_input1['SalePrice'] < 700000])
plt.figure(figsize=(8, 6))
sns.countplot(x='SaleCondition', data=_input1[_input1['SalePrice'] < 700000])
plt.figure(figsize=(8, 6))
sns.countplot(x='BldgType', data=_input1[_input1['SalePrice'] < 700000])
plt.figure(figsize=(10, 8))
sns.distplot(_input1['YearRemodAdd'])
plt.figure(figsize=(15, 15))
corr = _input1.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.figure(figsize=(10, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
sns.pointplot(x=_input1['OverallQual'], y=_input1['SalePrice'], ax=ax1)
sns.lineplot(x=_input1['OverallCond'], y=_input1['SalePrice'], ax=ax2)
sns.scatterplot(x=_input1.GrLivArea, y=_input1.SalePrice)
temp_y = y.copy()
temp_train = _input1.copy()
temp_test = _input0.copy()
temp_total = temp_train.append(temp_test)
print('temp_y shape', temp_y.shape)
print('temp_train shape', temp_train.shape)
print('temp_test shape', temp_test.shape)
print('temp_total shape', temp_total.shape)
temp_total.shape
sns.heatmap(temp_total.isnull())
print(temp_train['MiscFeature'].value_counts())
temp_total.loc[~temp_total['MiscFeature'].isnull(), 'MiscFeature'] = 1
temp_total.loc[temp_total['MiscFeature'].isnull(), 'MiscFeature'] = 0
print(temp_total['MiscFeature'].value_counts())
temp_total.loc[~temp_total['PoolQC'].isnull(), 'PoolQC'] = 1
temp_total.loc[temp_total['PoolQC'].isnull(), 'PoolQC'] = 0
print('Null PoolQC in trainig set:', temp_total['PoolQC'].isnull().sum())
temp_total.loc[~temp_total['Alley'].isnull(), 'Alley'] = 1
temp_total.loc[temp_total['Alley'].isnull(), 'Alley'] = 0
print('Null Alley in test set:', temp_total['Alley'].isnull().sum())
temp_total.loc[~temp_total['Fence'].isnull(), 'Fence'] = 1
temp_total.loc[temp_total['Fence'].isnull(), 'Fence'] = 0
print('Null Fence in trainig set:', temp_total['Fence'].isnull().sum())
temp_total = temp_total.drop('MiscFeature', axis=1, inplace=False)
temp_total = temp_total.drop('PoolQC', axis=1, inplace=False)
temp_total = temp_total.drop('Alley', axis=1, inplace=False)
temp_total.shape
s = temp_total.isnull().sum()
s[s > 0]
temp_total[temp_total['FireplaceQu'].isnull()]['Fireplaces'].value_counts()
temp_total.loc[temp_total['FireplaceQu'].isnull(), 'FireplaceQu'] = 'NA'
plt.figure(figsize=(10, 8))
sns.countplot(x='FireplaceQu', data=temp_total)
print('Null Fireplace in trainig set:', temp_total['FireplaceQu'].isnull().sum())
s = temp_total.isnull().sum()
s[s > 0]
temp_total[temp_total['GarageYrBlt'].isnull()]['GarageArea'].value_counts()
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageCars'] = 0
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageArea'] = 0.0
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageQual'] = 'NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageCond'] = 'NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageFinish'] = 'NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageType'] = 'NA'
temp_total.loc[temp_total['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0
s = temp_total.isnull().sum()
s[s > 0]
temp_total[temp_total['LotFrontage'].isnull()].head(10)
plt.figure(figsize=(20, 10))
ax1 = plt.subplot(3, 3, 1)
ax2 = plt.subplot(3, 3, 2)
ax3 = plt.subplot(3, 3, 3)
sns.countplot(x='MSZoning', data=temp_total[temp_total['LotFrontage'].isnull()], ax=ax1)
sns.countplot(x='Street', data=temp_total[temp_total['LotFrontage'].isnull()], ax=ax2)
sns.countplot(x='Utilities', data=temp_total[temp_total['LotFrontage'].isnull()], ax=ax3)
print(temp_train[temp_train['MSZoning'] == 'RL']['LotFrontage'].mean())
print(temp_train[temp_train['MSZoning'] == 'RL']['LotFrontage'].median())
print(temp_train[temp_train['Street'] == 'Pave']['LotFrontage'].mean())
print(temp_train[temp_train['Street'] == 'Pave']['LotFrontage'].median())
print(temp_train[temp_train['Utilities'] == 'AllPub']['LotFrontage'].mean())
print(temp_train[temp_train['Utilities'] == 'AllPub']['LotFrontage'].median())
print(temp_train[(temp_train['MSZoning'] == 'RL') & (temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['LotFrontage'].mean())
print(temp_train[(temp_train['MSZoning'] == 'RL') & (temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['LotFrontage'].median())
meanForLotFrontage = np.mean([72.0, 69.0, 69.0])
print(meanForLotFrontage)
temp_total.loc[temp_total['LotFrontage'].isnull(), 'LotFrontage'] = meanForLotFrontage
temp_total['LotFrontage'].isnull().sum()
temp_total[temp_total['BsmtQual'].isnull()]['TotalBsmtSF'].value_counts()
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtCond'] = 'NA'
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtExposure'] = 'NA'
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtFinType1'] = 'NA'
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtFinType2'] = 'NA'
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtFinSF1'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtFinSF2'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtUnfSF'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'TotalBsmtSF'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtFullBath'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtHalfBath'] = 0.0
temp_total.loc[temp_total['BsmtQual'].isnull(), 'BsmtQual'] = 'NA'
s = temp_total.isnull().sum()
s[s > 0]
print(temp_train[(temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['MasVnrArea'].mean())
print(temp_train[(temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['MasVnrArea'].median())
print(temp_train[(temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['MasVnrArea'].value_counts())
temp_train[(temp_train['Street'] == 'Pave') & (temp_train['Utilities'] == 'AllPub')]['MasVnrType'].value_counts()
temp_total.loc[temp_total['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
temp_total.loc[temp_total['MasVnrArea'].isnull(), 'MasVnrArea'] = 0.0
temp_total.loc[temp_total['MSZoning'].isnull(), 'MSZoning'] = 'RL'
temp_total.loc[temp_total['Utilities'].isnull(), 'Utilities'] = 'AllPub'
temp_total.loc[temp_total['Exterior1st'].isnull(), 'Exterior1st'] = 'VinylSd'
temp_total.loc[temp_total['Exterior2nd'].isnull(), 'Exterior2nd'] = 'VinylSd'
temp_total.loc[temp_total['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = 0.0
temp_total.loc[temp_total['BsmtFinSF2'].isnull(), 'BsmtFinSF2'] = 0.0
temp_total.loc[temp_total['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = 0.0
temp_total.loc[temp_total['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 0.0
temp_total.loc[temp_total['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
temp_total.loc[temp_total['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0.0
temp_total.loc[temp_total['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0.0
temp_total.loc[temp_total['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0.0
temp_total.loc[temp_total['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'
temp_total.loc[temp_total['Functional'].isnull(), 'Functional'] = 'Typ'
temp_total.loc[temp_total['GarageCars'].isnull(), 'GarageCars'] = 2.0
temp_total.loc[temp_total['GarageArea'].isnull(), 'GarageArea'] = 480.0
temp_total.loc[temp_total['SaleType'].isnull(), 'SaleType'] = 'WD'
s = temp_total.isnull().sum()
s[s > 0]
temp_total.loc[temp_total['BsmtCond'].isnull(), 'BsmtCond'] = 'TA'
temp_total.loc[temp_total['BsmtExposure'].isnull(), 'BsmtExposure'] = 'No'
temp_total.loc[temp_total['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'Unf'
s = temp_total.isnull().sum()
s[s > 0]
print(temp_total['GarageArea'].corr(temp_total['GarageCars']))
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
sns.distplot(temp_train['GarageArea'], ax=ax1)
sns.distplot(temp_train['GarageArea'] * temp_train['GarageCars'], ax=ax2)
temp_total['TotalArea'] = temp_total['TotalBsmtSF'] + temp_total['1stFlrSF'] + temp_total['2ndFlrSF'] + temp_total['GrLivArea'] + temp_total['GarageArea']
temp_total['Bathrooms'] = temp_total['FullBath'] + temp_total['HalfBath'] * 0.5 + temp_total['BsmtFullBath'] + 0.5 * temp_total['BsmtHalfBath']
print(temp_total.shape)
temp_total['Garage'] = temp_total['GarageArea'] * temp_total['GarageCars']
print(temp_total.shape)
temp_total.dtypes
temp_total.columns
convert_features = {'MSSubClass': str, 'OverallCond': str, 'OverallQual': str, 'SaleCondition': str}
temp_total = temp_total.astype(convert_features)
temp_total = pd.get_dummies(temp_total, drop_first=True)
temp_total.shape
temp_total = temp_total.drop('Id', axis=True, inplace=False)
plt.figure(figsize=(20, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
print('GrLivArea skew:', temp_total['GrLivArea'].skew())
print('GrLivArea kurtosis:', temp_total['GrLivArea'].kurtosis())
sns.distplot(temp_total['GrLivArea'], ax=ax1)
print('LotArea skew:', temp_total['LotArea'].skew())
print('LotArea kurtosis:', temp_total['LotArea'].kurtosis())
sns.distplot(temp_total['LotArea'], ax=ax2)
plt.figure(figsize=(20, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
print('1stFlrSF skew:', temp_total['1stFlrSF'].skew())
print('1stFlrSF kurtosis:', temp_total['1stFlrSF'].kurtosis())
sns.distplot(temp_total['1stFlrSF'], ax=ax1)
print('2ndFlrSF skew:', temp_total['1stFlrSF'].skew())
print('2ndFlrSF kurtosis:', temp_total['1stFlrSF'].kurtosis())
sns.distplot(temp_total['2ndFlrSF'], ax=ax2)
plt.figure(figsize=(20, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
print('TotalArea skew:', temp_total['TotalArea'].skew())
print('TotalArea kurtosis:', temp_total['TotalArea'].kurtosis())
sns.distplot(temp_total['TotalArea'], ax=ax1)
print('Garage skew:', temp_total['Garage'].skew())
print('Garage kurtosis:', temp_total['Garage'].kurtosis())
sns.distplot(temp_total['Garage'], ax=ax2)
temp_total['GrLivArea'] = np.log1p(temp_total['GrLivArea'])
temp_total['LotArea'] = np.log1p(temp_total['LotArea'])
temp_total['1stFlrSF'] = np.log1p(temp_total['1stFlrSF'])
temp_total['2ndFlrSF'] = np.log1p(temp_total['2ndFlrSF'])
temp_total['TotalArea'] = np.log1p(temp_total['TotalArea'])
temp_total['Garage'] = np.log1p(temp_total['Garage'])
temp_train = temp_total.iloc[:1460, :]
temp_test = temp_total.iloc[1460:, :]
print(temp_train.shape)
print(temp_test.shape)
temp_test = temp_test.drop('SalePrice', axis=1, inplace=False)
temp_train = temp_train[temp_train['SalePrice'] < 700000]
temp_y = temp_train['SalePrice']
temp_train = temp_train.drop('SalePrice', axis=1, inplace=False)
temp_y = np.log1p(temp_y)
print(temp_test.shape)
print(temp_train.shape)
sns.distplot(temp_y)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(temp_train, temp_y, test_size=0.2, random_state=42)
print('X_train size:', X_train.shape)
print('X_test size:', X_test.shape)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()