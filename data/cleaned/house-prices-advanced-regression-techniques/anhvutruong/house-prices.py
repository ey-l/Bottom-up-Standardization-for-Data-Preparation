import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head(5)
df_train.info()
lst = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

def converted_types(columns):
    for column in columns:
        df_train[column] = df_train[column].astype('category').cat.codes
converted_types(lst)
lst = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

def filled_values(columns):
    for column in columns:
        df_train[column].fillna(int(df_train[column].mean()), inplace=True)
filled_values(lst)
df_train.info()
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head(5)
df_test.info()
lst = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

def converted_types(columns):
    for column in columns:
        df_test[column] = df_test[column].astype('category').cat.codes
converted_types(lst)
lst = ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']

def filled_values(columns):
    for column in columns:
        df_test[column].fillna(int(df_test[column].mean()), inplace=True)
filled_values(lst)
df_test.info()
import matplotlib as mlp
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 5)
columns = df_train.select_dtypes(include=['int']).columns.tolist()
print(columns)
for column in columns[:]:
    if df_train[column].value_counts().shape[0] > 20:
        columns.remove(column)
for column in columns:
    sns.countplot(x=column, data=df_train)
    plt.title(column)

sns.histplot(x='SalePrice', kde=True, data=df_train)
plt.title('Sale Price')

x = df_train.drop(columns=['SalePrice'])
y = df_train.SalePrice
print('x Shape:', x.shape)
print('y Shape:', y.shape)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
print('x train Shape:', x_train.shape)
print('y train Shape:', y_train.shape)
print('x test Shape:', x_test.shape)
print('y test Shape:', y_test.shape)
from sklearn.linear_model import LinearRegression
linear = LinearRegression()