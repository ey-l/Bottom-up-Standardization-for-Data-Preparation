import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', na_values='?')
_input1.shape
_input1.head()
_input1.describe()
_input1.nunique()
_input1.info()
_input1 = _input1.drop_duplicates(keep='first', inplace=False)
print(_input1.shape)
_input1.isna().sum()
_input1.dtypes
train_df1 = _input1.drop(['Alley', 'Neighborhood', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities', 'ExterCond', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Electrical', 'GarageQual', 'PoolQC', 'MiscFeature'], axis=1)
train_df1.shape
train_df1['SalePrice'].value_counts()
train_df1.columns
train_df1.shape
sns.histplot(train_df1['SalePrice'], bins=20)
cat_cols = _input1.select_dtypes(include='object').columns
for col in cat_cols:
    _input1[col] = _input1[col].astype('category')
cat_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'BldgType', 'RoofStyle', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
num_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
X = train_df1.drop(['SalePrice'], axis=1)
y = train_df1['SalePrice']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train.value_counts(normalize=True) * 100
y_test.value_counts(normalize=True) * 100
df_cat_train = X_train[cat_cols]
df_cat_test = X_test[cat_cols]
print(df_cat_train.shape)
print(df_cat_test.shape)
df_num_train = X_train[num_cols]
df_num_test = X_test[num_cols]
print(df_num_train.shape)
print(df_num_test.shape)
from sklearn.impute import SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')