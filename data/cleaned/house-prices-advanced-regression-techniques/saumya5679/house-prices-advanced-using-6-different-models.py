import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1)
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
corr = train.drop('Id', 1).corr().sort_values(by='SalePrice', ascending=False).round(2)
print(corr['SalePrice'])
sns.scatterplot(x='OverallQual', y='SalePrice', data=train)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
train = train.drop(train.loc[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index, 0)
train.reset_index(drop=True, inplace=True)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=0.8, square=True)
cols = corr['SalePrice'].head(10).index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

cols = corr['SalePrice'].head(10).index
cm = np.corrcoef(train[cols].values.T)
cm
sns.pairplot(train[corr['SalePrice'].head(10).index])
trainrow = train.shape[0]
testrow = test.shape[0]
testids = test['Id'].copy()
y_train = train['SalePrice'].copy()
data = pd.concat((train, test)).reset_index(drop=True)
data = data.drop('SalePrice', 1)
data = data.drop('Id', axis=1)
data.head()
missing = data.isnull().sum().sort_values(ascending=False)
missing
missing = missing.drop(missing[missing == 0].index)
missing
for cols in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish']:
    data[cols].fillna('NA', inplace=True)
for bsmt in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    data[bsmt].fillna('NA', inplace=True)
for gar in ['GarageYrBlt', 'GarageType', 'GarageCars', 'GarageArea']:
    data[gar].fillna(0, inplace=True)
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)
data['MasVnrType'] = data['MasVnrType'].fillna('NA')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].dropna().sort_values().index[0])
data['Utilities'] = data['Utilities'].fillna(data['Utilities'].dropna().sort_values().index[0])
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
data['Functional'] = data['Functional'].fillna(data['Functional'].dropna().sort_values().index[0])
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)
data['Exterior2nd'] = data['Exterior2nd'].fillna('NA')
data['Exterior1st'] = data['Exterior1st'].fillna('NA')
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].dropna().sort_values().index[0])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].dropna().sort_values().index[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].dropna().sort_values().index[0])
missing = data.isnull().sum().sort_values(ascending=False)
missing = missing.drop(missing[missing == 0].index)
missing
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())
missing = data.isnull().sum().sort_values(ascending=False)
missing = missing.drop(missing[missing == 0].index)
missing
data['GrLivArea_2'] = data['GrLivArea'] ** 2
data['GrLivArea_3'] = data['GrLivArea'] ** 3
data['GrLivArea_4'] = data['GrLivArea'] ** 4
data['TotalBsmtSF_2'] = data['TotalBsmtSF'] ** 2
data['TotalBsmtSF_3'] = data['TotalBsmtSF'] ** 3
data['TotalBsmtSF_4'] = data['TotalBsmtSF'] ** 4
data['GarageCars_2'] = data['GarageCars'] ** 2
data['GarageCars_3'] = data['GarageCars'] ** 3
data['GarageCars_4'] = data['GarageCars'] ** 4
data['1stFlrSF_2'] = data['1stFlrSF'] ** 2
data['1stFlrSF_3'] = data['1stFlrSF'] ** 3
data['1stFlrSF_4'] = data['1stFlrSF'] ** 4
data['GarageArea_2'] = data['GarageArea'] ** 2
data['GarageArea_3'] = data['GarageArea'] ** 3
data['GarageArea_4'] = data['GarageArea'] ** 4
data['Floorfeet'] = data['1stFlrSF'] + data['2ndFlrSF']
data = data.drop(['1stFlrSF', '2ndFlrSF'], 1)
data = pd.get_dummies(data=data, columns=['MSSubClass'], prefix='MSSubClass')
data = pd.get_dummies(data=data, columns=['MSZoning'], prefix='MSZoning')
data.head()
data = pd.get_dummies(data=data, columns=['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle'])
data.head()
data = pd.get_dummies(data=data, columns=['OverallQual'], prefix='OverallQual')
data = pd.get_dummies(data=data, columns=['OverallCond'], prefix='OverallCond')
data['Remodeled'] = 0
data.loc[data['YearBuilt'] != data['YearRemodAdd'], 'Remodeled'] = 1
data = data.drop('YearRemodAdd', 1)
data = pd.get_dummies(data=data, columns=['Remodeled'])
data = pd.get_dummies(data=data, columns=['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'])
data['Bath'] = data['BsmtFullBath'] + data['BsmtHalfBath'] * 0.5 + data['FullBath'] + data['HalfBath'] * 0.5
data = data.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], 1)
data = pd.get_dummies(data=data, columns=['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'])
data.loc[data['GarageYrBlt'] == 2207.0, 'GarageYrBlt'] = 0
from sklearn.preprocessing import StandardScaler
x_train = data.iloc[:trainrow]
x_test = data.iloc[trainrow:]
scaler = StandardScaler()