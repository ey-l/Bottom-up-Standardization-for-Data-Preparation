import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine_data = pd.concat([_input1, _input0])
combine_data.shape
combine_data.tail()
top_null_columns = combine_data.isnull().sum() / combine_data.shape[0] * 100
deleted_null_columns = top_null_columns[top_null_columns.values > 20].keys()
deleted_null_columns
af_del_combine = combine_data.drop(deleted_null_columns, axis='columns')
af_del_combine.head()
plt.figure(figsize=(20, 7))
sns.heatmap(af_del_combine.isnull())
numeric_data = af_del_combine.select_dtypes(['int64', 'float64'])
numeric_data_columns = numeric_data.columns
numeric_data_columns
numeric_null = numeric_data.isnull().sum()
numeric_null_collumns = numeric_null[numeric_null.values > 0].keys()
numeric_null_collumns
numeric_null
numeric_data[numeric_data[numeric_null_collumns].isnull().any(axis=1)].head()
numeric_fill_data_mean = numeric_data.fillna(numeric_data.mean())
numeric_fill_data_mean.isnull().sum().sum()
categorical_data = af_del_combine.select_dtypes(['O'])
categorical_data.head()
cat_null_col = categorical_data.isnull().sum()
cat_null_col = cat_null_col[cat_null_col.values > 0].keys()
cat_null_col
categorical_data[categorical_data[cat_null_col].isnull().any(axis=1)].head()
categorical_fill_data_mode = categorical_data.copy()
categorical_fill_data_mode.head()
for column in cat_null_col:
    print(column, categorical_data[column].mode()[0])
    categorical_fill_data_mode[column] = categorical_data[column].fillna(categorical_data[column].mode()[0])
categorical_data.isnull().sum().sum()
categorical_fill_data_mode.isnull().sum().sum()
combine_data.head()
combine_data.isnull().sum()
numeric_fill_data_mean.head()
categorical_fill_data_mode.head()
combine_fill_data = pd.concat([numeric_fill_data_mean, categorical_fill_data_mode], axis=1, sort=False)
combine_fill_data.head()
combine_fill_data.isnull().sum().sum()
categorical_columns = categorical_data.columns
categorical_columns
combine_fill_data[categorical_columns].head()
combine_fill_data['GarageCond'].unique()
ordinal_cat_columns = ['KitchenQual', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'GarageQual', 'GarageCond']
nominal_cat_columns = ['BldgType', 'Condition1', 'Condition2', 'Exterior1st', 'Electrical', 'GarageType', 'GarageFinish', 'SaleCondition', 'RoofMatl', 'SaleType', 'MasVnrType', 'LandSlope', 'LotShape', 'PavedDrive', 'Utilities', 'Heating', 'Functional', 'LandContour', 'LotConfig', 'Exterior2nd', 'Neighborhood', 'HouseStyle', 'Street', 'MSZoning', 'Foundation', 'RoofStyle', 'CentralAir']
len(nominal_cat_columns)
len(ordinal_cat_columns)
combine_fill_data['GarageCond'].unique()
combine_map_data = combine_fill_data.copy()
KitchenQual_map = {'Gd': 3, 'TA': 2, 'Ex': 4, 'Fa': 1}
ExterQual_map = {'Gd': 3, 'TA': 2, 'Ex': 4, 'Fa': 1}
ExterCond_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
BsmtQual_map = {'Gd': 3, 'TA': 2, 'Ex': 4, 'Fa': 1}
BsmtCond_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1}
BsmtExposure_map = {'No': 1, 'Gd': 4, 'Mn': 2, 'Av': 3}
BsmtFinType1_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
BsmtFinType2_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
HeatingQC_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
GarageQual_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
GarageCond_map = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1, 'Ex': 5}
combine_map_data['KitchenQual'] = combine_fill_data['KitchenQual'].map(KitchenQual_map)
combine_map_data['ExterQual'] = combine_fill_data['ExterQual'].map(ExterQual_map)
combine_map_data['ExterCond'] = combine_fill_data['ExterCond'].map(ExterCond_map)
combine_map_data['BsmtQual'] = combine_fill_data['BsmtQual'].map(BsmtQual_map)
combine_map_data['BsmtCond'] = combine_fill_data['BsmtCond'].map(BsmtCond_map)
combine_map_data['BsmtExposure'] = combine_fill_data['BsmtExposure'].map(BsmtExposure_map)
combine_map_data['BsmtFinType1'] = combine_fill_data['BsmtFinType1'].map(BsmtFinType1_map)
combine_map_data['BsmtFinType2'] = combine_fill_data['BsmtFinType2'].map(BsmtFinType2_map)
combine_map_data['HeatingQC'] = combine_fill_data['HeatingQC'].map(HeatingQC_map)
combine_map_data['GarageQual'] = combine_fill_data['GarageQual'].map(GarageQual_map)
combine_map_data['GarageCond'] = combine_fill_data['GarageCond'].map(GarageCond_map)
combine_map_data['GarageCond']
combine_map_data.head()
combine_dummy_data = pd.get_dummies(combine_map_data, drop_first=True)
combine_dummy_data.head()
combine_dummy_data[combine_dummy_data['CentralAir_Y'] == 0].head()
from sklearn.preprocessing import RobustScaler
saleprice = np.log(_input1['SalePrice'] + 1)
saleprice
combine_dummy_drop_data = combine_dummy_data.drop(['Id'], axis=1)
combine_dummy_drop_data
robust = RobustScaler()