import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as nm
import matplotlib
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.tail()
X1 = _input0
id = X1.Id
datas = _input1
y = datas.SalePrice
datas = datas.drop(['SalePrice'], axis='columns')
datas = pd.concat([datas, X1], axis=0)
datas.shape
datas.shape
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
datas.isna().sum()
datas = datas.drop(['Alley'], axis='columns')
datas.isna().sum()
datas.shape
datas.tail()
X1.tail()
datas = datas.drop(['PoolQC'], axis='columns')
datas = datas.drop(['MiscFeature'], axis='columns')
datas.isna().sum()
cols_to_fill_zero1 = ['LotFrontage', 'Fence', 'MasVnrArea', 'Electrical', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageCars', 'GarageArea', 'SaleType']
datas[cols_to_fill_zero1] = datas[cols_to_fill_zero1].fillna(0)
datas.nunique()
datas.head()
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
datas.shape
datas.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
datas.shape
nan_cols = [i for i in datas.columns if datas[i].isnull().any()]
nan_cols
dataset = pd.get_dummies(datas, drop_first=True)
dataset.shape
dataset.head(1500)
y.shape
dataset.tail(1460)
X = dataset.iloc[:1460, :]
test = dataset.iloc[1460:, :]
X.shape
test.head()
from sklearn.model_selection import train_test_split
(train_X, test_X, train_y, test_y) = train_test_split(X, y, test_size=0.2, random_state=42)
X.shape
from sklearn.linear_model import LinearRegression