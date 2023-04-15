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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train
df_train.describe()
df_train.info()
print(df_train.isna().sum())
df_train.columns
df_train.isna().sum()
df_train.isnull()
plt.figure(figsize=(25, 16))
sns.heatmap(df_train.corr(), annot=True)

x = df_train.SalePrice.values
plt.plot(x, '.', color='g')
df_train = df_train[df_train['SalePrice'] < 700000]
x = df_train.SalePrice.values
plt.plot(x, '.', color='g')
sns.histplot(df_train.SalePrice)
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], height=2.5)

full = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
nan_data = pd.concat([full, percent], axis=1, keys=['Total', 'Percent'])
nan_data.head(20)
df_train = df_train.drop(nan_data[nan_data['Total'] > 100].index, 1)
df_train = df_train.fillna(method='bfill')
if df_train.isna().sum().max() > 0:
    df_train = df_train.fillna(method='ffill')
df_train.isna().sum().max()
df_train.info()
if 'BrkFace' in df_train.columns:
    print('ok')
else:
    print(0)
encode = OrdinalEncoder()
obj = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition', 'MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'GarageType', 'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageQual', 'GarageCond', 'BsmtCond']
df_train[obj] = encode.fit_transform(df_train[obj])
df_train.info()
x_train = df_train.drop('SalePrice', axis=1).values
y_train = df_train['SalePrice'].values
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_test
df_test.isna().sum().max()
full = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum() / df_test.isnull().count()).sort_values(ascending=False)
nan_data = pd.concat([full, percent], axis=1, keys=['Overall', 'Percent'])
nan_data
df_test = df_test.fillna(method='bfill')
df_test = df_test.fillna(method='ffill')
df_test.isna().sum().max()
df_test.isna().sum().max()
encode = OrdinalEncoder()
column = list(df_test.columns)
obj = []
v = []
for i in column:
    if type(df_test[i].values[1]) == str:
        obj.append(i)
df_test[obj] = encode.fit_transform(df_test[obj])
df_test.info()
ID = df_test['Id'].values
ID
df_test = df_test[['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']]
x_test = df_test.values
df_test.info()
model = LinearRegression()