import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.drop(columns=['FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtExposure', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'HeatingQC', 'KitchenQual', 'Functional', 'SaleType', 'SaleCondition', 'ExterQual', 'Street', 'LotArea', 'LotFrontage', 'MSZoning', 'MSSubClass', 'LotShape', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BldgType', 'PoolArea', 'MiscVal', 'Alley', 'FireplaceQu'], inplace=True)
df
df.replace({'LandContour': {'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3}, 'Utilities': {'AllPub': 0, 'NoSeWa': 1}, 'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2}, 'ExterCond': {'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4}, 'Foundation': {'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}, 'CentralAir': {'Y': 0, 'N': 1}, 'MasVnrType': {'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3}, 'Electrical': {'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4}, 'PavedDrive': {'Y': 0, 'N': 1, 'P': 2}}, inplace=True)
df['Electrical'].value_counts()
for i in df.columns:
    print(i, df[i].isnull().sum())
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df.isnull().sum()
len(df.columns)
x = df.drop(columns=['SalePrice'])
y = df['SalePrice']
df
from sklearn.linear_model import Lasso
model_2 = Lasso(normalize=True)
(X_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)