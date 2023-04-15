import pandas as pd
pd.get_option('display.max_columns')
pd.set_option('display.max_columns', 81)
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()
train_num_df = train_df.select_dtypes(include=['int64', 'float64'])
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.subplots(figsize=(12, 36))
i = 0
for (j, feature) in enumerate(train_num_df.columns):
    if feature not in ['Id', 'SalePrice']:
        i += 1
        plt.subplot(13, 3, i)
        sns.histplot(train_df[feature], kde=True)
        plt.tight_layout()
fig = plt.subplots(figsize=(12, 36))
i = 0
for (j, feature) in enumerate(train_num_df.columns):
    if feature not in ['Id', 'SalePrice']:
        i += 1
        plt.subplot(13, 3, i)
        sns.scatterplot(x=train_df[feature], y=train_df['SalePrice'])
        plt.tight_layout()
fig = plt.subplots(figsize=(12, 15))
for (i, feature) in enumerate(['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces']):
    plt.subplot(6, 3, i + 1)
    sns.barplot(x=train_df[feature], y=train_df['SalePrice'])
    plt.tight_layout()
train_cat_df = train_df.select_dtypes(include=['object'])
fig = plt.subplots(figsize=(12, 60))
for (i, feature) in enumerate(train_cat_df.columns):
    plt.subplot(15, 3, i + 1)
    sns.boxplot(x=train_df['SalePrice'], y=train_df[feature])
    plt.tight_layout()
plt.figure(figsize=(20, 16))
sns.heatmap(train_num_df.corr(), annot=True)

import numpy as np
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
y_train = train_df['SalePrice']
train_df = train_df.drop(['Id', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'SalePrice'], axis=1)
test_df = test_df.drop(['Id', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1)
train_df.isnull().sum()
test_df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
features_to_encode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for feature in features_to_encode:
    train_df[feature].fillna('miss', inplace=True)
    le = LabelEncoder()