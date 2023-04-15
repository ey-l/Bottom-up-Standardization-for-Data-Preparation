import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data.sample(5)
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test.head()
data.info()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def func_na(x, temp):
    if pd.isna(x):
        return np.random.choice(temp)
    else:
        return x

def data_preprocessing(data, flag):
    numerical_column = [column for column in data.columns if data[column].dtypes == 'int64' or data[column].dtypes == 'float64']
    categorical_column = [column for column in data.columns if data[column].dtypes == 'object']
    num_df = data[numerical_column]
    cat_df = data[categorical_column]
    if flag == 'train':
        sales_price_label = num_df['SalePrice']
        num_df['YearsOld'] = num_df.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)
        num_df.drop(['YrSold', 'YearBuilt', 'Id', 'SalePrice'], axis=1, inplace=True)
    else:
        num_df['YearsOld'] = num_df.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)
        num_df.drop(['YrSold', 'YearBuilt', 'Id'], axis=1, inplace=True)
    OneHotEncoding_columns = ['LotShape', 'MSZoning', 'Street', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 'Functional', 'Electrical', 'MasVnrType', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    label_encoding = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    replace = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    delete_column = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
    fill_previous_value = label_encoding + OneHotEncoding_columns
    cat_df.drop(delete_column, inplace=True, axis=1)
    for nan_column in fill_previous_value:
        temp = cat_df[nan_column].unique().copy()
        temp = temp[~pd.isna(temp)]
        cat_df[nan_column] = cat_df.apply(lambda x: func_na(x[nan_column], temp), axis=1)
    one_hot_df = cat_df[OneHotEncoding_columns]
    label_encod_df = cat_df[label_encoding]
    encod = OneHotEncoder()