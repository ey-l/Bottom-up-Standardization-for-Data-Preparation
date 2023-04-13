import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_ID = _input1['Id']
test_ID = _input0['Id']
y_train = y = _input1['SalePrice']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1.select_dtypes(include=['int64', 'float64'])
_input1.select_dtypes(include=['object'])
categorical = len(_input1.select_dtypes(include=['object']).columns)
numbers = len(_input1.select_dtypes(include=['float64', 'int64']).columns)
print('Total number of Categorical Data is:', categorical)
print('Total number of Numerical Data is:', numbers)
print('Total Features are:', categorical + numbers)
_input1.shape
_input0.shape
plt.figure(figsize=(10, 5))
sns.distplot(_input1['SalePrice'], color='salmon')
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
c = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[c].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=c.values, xticklabels=c.values)
most_cor = pd.DataFrame(c)
most_cor
sns.jointplot(x=_input1['OverallQual'], y=_input1['SalePrice'], kind='reg', color='skyblue', height=7)
sns.jointplot(x=_input1['GrLivArea'], y=_input1['SalePrice'], kind='hex', color='violet', height=7)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index).reset_index(drop=True)
sns.jointplot(x=_input1['GrLivArea'], y=_input1['SalePrice'], kind='hex', color='violet', height=7)
sns.boxplot(x=_input1['GarageCars'], y=_input1['SalePrice'])
_input1 = _input1.drop(_input1[(_input1['GarageCars'] > 3) & (_input1['SalePrice'] < 300000)].index).reset_index(drop=True)
sns.boxplot(x=_input1['GarageCars'], y=_input1['SalePrice'])
sns.jointplot(x=_input1['GarageArea'], y=_input1['SalePrice'], kind='reg')
sns.jointplot(x=_input1['GarageArea'], y=_input1['SalePrice'], kind='reg', color='coral', height=7)
sns.jointplot(x=_input1['1stFlrSF'], y=_input1['SalePrice'], kind='hex', color='gold', height=7)
sns.boxplot(x=_input1['TotRmsAbvGrd'], y=_input1['SalePrice'])
sns.jointplot(x=_input1['YearBuilt'], y=_input1['SalePrice'], kind='reg', color='green', height=7)
ntrain = _input1.shape[0]
ntest = _input0.shape[0]
y_train = _input1.SalePrice.values
total = pd.concat((_input1, _input0)).reset_index(drop=True)
total = total.drop(['SalePrice'], axis=1, inplace=False)
print('Combined dataset size is : ', total.shape)
totalnull = total.isnull().sum() / len(total) * 100
totalnull = totalnull.drop(totalnull[totalnull == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Values': totalnull})
missing_data
(f, ax) = plt.subplots(figsize=(13, 5))
plt.xticks(rotation='90')
sns.barplot(x=totalnull.index, y=totalnull)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
total['PoolQC'] = total['PoolQC'].fillna('None')
total['MiscFeature'] = total['MiscFeature'].fillna('None')
total['Alley'] = total['Alley'].fillna('None')
total['Fence'] = total['Fence'].fillna('None')
total['FireplaceQu'] = total['FireplaceQu'].fillna('None')
lot = total.groupby('Neighborhood')['LotFrontage']
print(lot.median())
total.loc[total.LotFrontage.isnull(), 'LotFrontage'] = total.groupby('Neighborhood').LotFrontage.transform('median')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    total[col] = total[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    total[col] = total[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    total[col] = total[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    total[col] = total[col].fillna('None')
total['MasVnrType'] = total['MasVnrType'].fillna('None')
total['MasVnrArea'] = total['MasVnrArea'].fillna(0)
total['MSZoning'] = total['MSZoning'].fillna(total['MSZoning'].mode()[0])
total['Functional'] = total['Functional'].fillna('Typ')
total['Electrical'] = total['Electrical'].fillna('SBrkr')
total['KitchenQual'] = total['KitchenQual'].fillna('TA')
total['Exterior1st'] = total['Exterior1st'].fillna(total['Exterior1st'].mode()[0])
total['Exterior2nd'] = total['Exterior2nd'].fillna(total['Exterior2nd'].mode()[0])
total['SaleType'] = total['SaleType'].fillna(total['SaleType'].mode()[0])
total['MSSubClass'] = total['MSSubClass'].fillna('None')
total['MSSubClass'] = total['MSSubClass'].apply(str)
total['OverallCond'] = total['OverallCond'].astype(str)
total['YrSold'] = total['YrSold'].astype(str)
total['MoSold'] = total['MoSold'].astype(str)
total['TotalSF'] = total['TotalBsmtSF'] + total['1stFlrSF'] + total['2ndFlrSF']
total['Bathrooms'] = total['BsmtHalfBath'] + total['BsmtFullBath'] + total['HalfBath'] + total['FullBath']
total['TotalSqu'] = total['BsmtFinSF1'] + total['BsmtFinSF2'] + total['1stFlrSF'] + total['2ndFlrSF']
total['pool'] = total['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
total['2ndfloor'] = total['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
total['garage'] = total['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
total['Basement'] = total['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
total['Fireplace'] = total['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
total = total.drop(['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd'], axis=1, inplace=False)
total = total.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
missing = total.isnull().sum()
missing
total.select_dtypes(include=['object']).columns
from sklearn.preprocessing import LabelEncoder
c = ('Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Electrical', 'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSSubClass', 'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'YrSold')
for i in c:
    l = LabelEncoder()