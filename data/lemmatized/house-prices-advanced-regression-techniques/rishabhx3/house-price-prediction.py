import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train Shape: ', _input1.shape)
print('Test Shape: ', _input0.shape)
print(_input1.info())
print(_input0.info())
print(_input1.isnull().sum())
sns.heatmap(_input1.isnull())
print(_input0.isnull().sum())
sns.heatmap(_input0.isnull())
cat_col_train = ['FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageQual', 'GarageCond']
ncat_col_train = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
for i in cat_col_train:
    _input1[i] = _input1[i].fillna(_input1[i].mode()[0])
for j in ncat_col_train:
    _input1[j] = _input1[j].fillna(_input1[j].mean())
cat_col_test = ['FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageQual', 'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
ncat_col_test = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
for i in cat_col_test:
    _input0[i] = _input0[i].fillna(_input0[i].mode()[0])
for j in ncat_col_test:
    _input0[j] = _input0[j].fillna(_input0[j].mean())
to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
for k in to_drop:
    _input1 = _input1.drop([k], axis=1, inplace=False)
    _input0 = _input0.drop([k], axis=1, inplace=False)
sns.heatmap(_input1.isnull())
sns.heatmap(_input0.isnull())
print('Train Shape: ', _input1.shape)
print('Test Shape: ', _input0.shape)
final_df = pd.concat([_input1, _input0], axis=0)
final_df.shape
all_cat_col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

def cat_onehot_encoding(multicol):
    df_final = final_df
    i = 0
    for fields in multicol:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df = final_df.drop([fields], axis=1, inplace=False)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
final_df = cat_onehot_encoding(all_cat_col)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
df_train = final_df.iloc[:1460, :]
df_test = final_df.iloc[1460:, :]
df_test = df_test.drop(['SalePrice'], axis=1, inplace=False)
print('Train Shape: ', df_train.shape)
print('Test Shape: ', df_test.shape)
x_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']
import xgboost
xgb_model = xgboost.XGBRegressor()