import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_ID = train['Id']
test_ID = test['Id']
y_train = y = train['SalePrice']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train.select_dtypes(include=['int64', 'float64'])
train.select_dtypes(include=['object'])
categorical = len(train.select_dtypes(include=['object']).columns)
numbers = len(train.select_dtypes(include=['float64', 'int64']).columns)
print('Total number of Categorical Data is:', categorical)
print('Total number of Numerical Data is:', numbers)
print('Total Features are:', categorical + numbers)
train.shape
test.shape
plt.figure(figsize=(10, 5))
sns.distplot(train['SalePrice'], color='salmon')
corrmat = train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
c = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[c].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=c.values, xticklabels=c.values)

most_cor = pd.DataFrame(c)
most_cor
sns.jointplot(x=train['OverallQual'], y=train['SalePrice'], kind='reg', color='skyblue', height=7)
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='hex', color='violet', height=7)
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='hex', color='violet', height=7)
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
train = train.drop(train[(train['GarageCars'] > 3) & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg', color='coral', height=7)
sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='hex', color='gold', height=7)
sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'])
sns.jointplot(x=train['YearBuilt'], y=train['SalePrice'], kind='reg', color='green', height=7)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
total = pd.concat((train, test)).reset_index(drop=True)
total.drop(['SalePrice'], axis=1, inplace=True)
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
total.drop(['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd'], axis=1, inplace=True)
total = total.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
missing = total.isnull().sum()
missing
total.select_dtypes(include=['object']).columns
from sklearn.preprocessing import LabelEncoder
c = ('Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Electrical', 'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSSubClass', 'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'YrSold')
for i in c:
    l = LabelEncoder()