import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
PATH = '_data/input/house-prices-advanced-regression-techniques/'
train_df = pd.read_csv(PATH + 'train.csv')
test_df = pd.read_csv(PATH + 'test.csv')
sub_df = pd.read_csv(PATH + 'sample_submission.csv')
print('train_df shape :: ', train_df.shape)
print('test_df shape :: ', test_df.shape)
train_df.head()
grouped = train_df['YearBuilt'].groupby(train_df['MSZoning'])
print(grouped.mean())
grouped = train_df['SalePrice'].groupby(train_df['MSZoning'])
print(grouped.mean())
for col in train_df.columns:
    if col not in test_df.columns:
        print(col, 'is the target column')
sns.distplot(train_df['SalePrice'])
sns.distplot(train_df['OverallQual'])
sns.distplot(test_df['OverallQual'])
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        print('[[', col, ']] column has >>', train_df[col].isnull().sum(), '<< nulls\n')
null_dominant_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
train_df.drop(columns=null_dominant_cols, inplace=True)
test_df.drop(columns=null_dominant_cols, inplace=True)

def preprocess(df):
    categorical_cols = ['LotConfig', 'LotArea', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'YearRemodAdd', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Utilities', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleCondition', 'MSZoning', 'LotShape', 'Street', 'LandContour']
    df['MasVnrType'].fillna('None', inplace=True)
    df['BsmtQual'].fillna('None', inplace=True)
    df['BsmtCond'].fillna('None', inplace=True)
    df['BsmtExposure'].fillna('None', inplace=True)
    df['BsmtFinType1'].fillna('None', inplace=True)
    df['BsmtFinType2'].fillna('None', inplace=True)
    df['Electrical'].fillna('None', inplace=True)
    df['GarageType'].fillna('None', inplace=True)
    df['GarageFinish'].fillna('None', inplace=True)
    df['GarageQual'].fillna('None', inplace=True)
    df['GarageCond'].fillna('None', inplace=True)
    df['Exterior1st'].fillna('None', inplace=True)
    df['Exterior2nd'].fillna('None', inplace=True)
    df['Utilities'].fillna('None', inplace=True)
    df['Electrical'].fillna('None', inplace=True)
    df['KitchenQual'].fillna('None', inplace=True)
    df['Functional'].fillna('None', inplace=True)
    df['SaleType'].fillna('None', inplace=True)
    df['MSZoning'].fillna('None', inplace=True)
    df['LotArea'] = np.log1p(df['LotArea'])
    df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    df['MasVnrArea'] = df['MasVnrArea'].astype(int)
    return df
train_df = preprocess(train_df)
test_df = preprocess(test_df)
print('former shape: ', train_df.shape, ' test : ', test_df.shape)
train_df.drop_duplicates()
test_df.drop_duplicates()
print('latter shape: ', train_df.shape, ' test : ', test_df.shape)
train_corrmatrix = train_df.corr()
cols = train_corrmatrix.nlargest(40, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
plt.figure(figsize=(50, 50))
sns.set(font_scale=2)
sns.heatmap(cm, cbar=True, linewidths=2, vmax=0.9, square=True, annot=True, fmt='.2f', annot_kws={'size': 17}, yticklabels=cols.values, xticklabels=cols.values)

cols = list(cols)
cols.extend(['YrSold'])
test_cols = []
for col in cols:
    if col != 'SalePrice':
        test_cols.append(col)
print('Choosed features : ', cols)
train_df = train_df[cols]
test_df = test_df[test_cols]

def add_features(df):
    df['house_age1'] = df['YrSold'] - df['YearBuilt']
    df['house_age2'] = df['YrSold'] - df['YearRemodAdd']
    df['garage_age'] = df['YrSold'] - df['GarageYrBlt']
    df['total_area'] = np.log1p(df['GrLivArea'] + df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])
    df['num_rooms'] = df['TotRmsAbvGrd'] + df['BedroomAbvGr'] + df['FullBath']
    return df
train_df = add_features(train_df)
test_df = add_features(test_df)
train_df['YrSold'] = train_df['YrSold'].replace({2008: 2, 2007: 1, 2006: 0, 2009: 3, 2010: 4})
test_df['YrSold'] = test_df['YrSold'].replace({2008: 2, 2007: 1, 2006: 0, 2009: 3, 2010: 4})
train_df.head()
sns.distplot(train_df['SalePrice'])
train_label = np.log1p(train_df['SalePrice'])
train_df.drop(columns=['SalePrice'], inplace=True)
train_df.head()
xgb_param = {'learning_rate': 0.03, 'max_depth': 40, 'verbosity': 3, 'nthread': 5, 'random_state': 0, 'subsample': 0.7, 'n_estimators': 5000, 'colsample_bytree': 0.8}
model_xgb = xgb.XGBRegressor(learning_rate=xgb_param['learning_rate'], max_depth=xgb_param['max_depth'], verbosity=xgb_param['verbosity'], nthread=xgb_param['nthread'], random_state=xgb_param['random_state'], subsample=xgb_param['subsample'], n_estimators=xgb_param['n_estimators'], colsample_bytree=xgb_param['colsample_bytree'])