import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.describe()
_input1.info()
print(_input1.isna().sum())
_input1.columns
_input1.isna().sum()
_input1.isnull()
plt.figure(figsize=(25, 16))
sns.heatmap(_input1.corr(), annot=True)
x = _input1.SalePrice.values
plt.plot(x, '.', color='g')
_input1 = _input1[_input1['SalePrice'] < 700000]
x = _input1.SalePrice.values
plt.plot(x, '.', color='g')
sns.histplot(_input1.SalePrice)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], height=2.5)
full = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
nan_data = pd.concat([full, percent], axis=1, keys=['Total', 'Percent'])
nan_data.head(20)
_input1 = _input1.drop(nan_data[nan_data['Total'] > 100].index, 1)
_input1 = _input1.fillna(method='bfill')
if _input1.isna().sum().max() > 0:
    _input1 = _input1.fillna(method='ffill')
_input1.isna().sum().max()
_input1.info()
if 'BrkFace' in _input1.columns:
    print('ok')
else:
    print(0)
encode = OrdinalEncoder()
obj = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition', 'MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'GarageType', 'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageQual', 'GarageCond', 'BsmtCond']
_input1[obj] = encode.fit_transform(_input1[obj])
_input1.info()
x_train = _input1.drop('SalePrice', axis=1).values
y_train = _input1['SalePrice'].values
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
_input0.isna().sum().max()
full = _input0.isnull().sum().sort_values(ascending=False)
percent = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
nan_data = pd.concat([full, percent], axis=1, keys=['Overall', 'Percent'])
nan_data
_input0 = _input0.fillna(method='bfill')
_input0 = _input0.fillna(method='ffill')
_input0.isna().sum().max()
_input0.isna().sum().max()
encode = OrdinalEncoder()
column = list(_input0.columns)
obj = []
v = []
for i in column:
    if type(_input0[i].values[1]) == str:
        obj.append(i)
_input0[obj] = encode.fit_transform(_input0[obj])
_input0.info()
ID = _input0['Id'].values
ID
_input0 = _input0[['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']]
x_test = _input0.values
_input0.info()
model = LinearRegression()