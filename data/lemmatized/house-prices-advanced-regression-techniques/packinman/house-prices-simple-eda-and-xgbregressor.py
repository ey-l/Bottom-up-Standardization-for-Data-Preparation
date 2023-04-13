import pandas as pd
pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 81)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
train_num_df = _input1.select_dtypes(include=['int64', 'float64'])
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.subplots(figsize=(12, 36))
i = 0
for (j, feature) in enumerate(train_num_df.columns):
    if feature not in ['Id', 'SalePrice']:
        i += 1
        plt.subplot(13, 3, i)
        sns.histplot(_input1[feature], kde=True)
        plt.tight_layout()
fig = plt.subplots(figsize=(12, 36))
i = 0
for (j, feature) in enumerate(train_num_df.columns):
    if feature not in ['Id', 'SalePrice']:
        i += 1
        plt.subplot(13, 3, i)
        sns.scatterplot(x=_input1[feature], y=_input1['SalePrice'])
        plt.tight_layout()
fig = plt.subplots(figsize=(12, 15))
for (i, feature) in enumerate(['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces']):
    plt.subplot(6, 3, i + 1)
    sns.barplot(x=_input1[feature], y=_input1['SalePrice'])
    plt.tight_layout()
train_cat_df = _input1.select_dtypes(include=['object'])
fig = plt.subplots(figsize=(12, 60))
for (i, feature) in enumerate(train_cat_df.columns):
    plt.subplot(15, 3, i + 1)
    sns.boxplot(x=_input1['SalePrice'], y=_input1[feature])
    plt.tight_layout()
plt.figure(figsize=(20, 16))
sns.heatmap(train_num_df.corr(), annot=True)
import numpy as np
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
y_train = _input1['SalePrice']
_input1 = _input1.drop(['Id', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'SalePrice'], axis=1)
_input0 = _input0.drop(['Id', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1)
_input1.isnull().sum()
_input0.isnull().sum()
from sklearn.preprocessing import LabelEncoder
features_to_encode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for feature in features_to_encode:
    _input1[feature] = _input1[feature].fillna('miss', inplace=False)
    le = LabelEncoder()