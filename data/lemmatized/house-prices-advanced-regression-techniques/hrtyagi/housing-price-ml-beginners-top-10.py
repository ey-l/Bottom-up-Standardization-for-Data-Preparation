import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
train_Id = _input1['Id']
test_Id = _input0['Id']
_input1.describe()
_input0.head()
_input0.describe()
sns.heatmap(_input1.isnull())
sns.heatmap(_input0.isnull())
_input1.info()
_input0.info()
_input1.isnull().sum().sort_values(ascending=False)[0:20]
_input0.isnull().sum().sort_values(ascending=False)[0:35]
list_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt']
for col in list_drop:
    del _input1[col]
    del _input0[col]
_input1.isnull().sum().sort_values(ascending=False)[0:15]
_input0.isnull().sum().sort_values(ascending=False)[0:30]
_input1.LotFrontage.value_counts(dropna=False)
_input1.LotFrontage = _input1.LotFrontage.fillna(_input1.LotFrontage.mean(), inplace=False)
_input0.LotFrontage = _input0.LotFrontage.fillna(_input0.LotFrontage.mean(), inplace=False)
print(_input1.BsmtCond.value_counts(dropna=False))
print(_input0.BsmtCond.value_counts(dropna=False))
list_fill_train = ['BsmtCond', 'BsmtQual', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'BsmtFinType2', 'BsmtExposure', 'FireplaceQu', 'MasVnrArea']
for j in list_fill_train:
    _input1[j] = _input1[j].fillna(_input1[j].mode()[0])
    _input0[j] = _input0[j].fillna(_input1[j].mode()[0])
print(_input1.isnull().sum().sort_values(ascending=False)[0:5])
print(_input0.isnull().sum().sort_values(ascending=False)[0:20])
_input1 = _input1.dropna(inplace=False)
_input1.shape
list_test_str = ['BsmtFinType1', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'MSZoning']
list_test_num = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']
for item in list_test_str:
    _input0[item] = _input0[item].fillna(_input0[item].mode()[0])
for item in list_test_num:
    _input0[item] = _input0[item].fillna(_input0[item].mean())
print(_input1.isnull().sum().sort_values(ascending=False)[0:5])
print(_input0.isnull().sum().sort_values(ascending=False)[0:5])
_input0.shape
Y = _input1['SalePrice']
del _input1['Id']
del _input0['Id']
del _input1['SalePrice']
_input0.shape
_input1.shape
final = pd.concat([_input1, _input0], axis=0)
final.shape
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

def One_hot_encoding(columns):
    final_df = final
    i = 0
    for fields in columns:
        df1 = pd.get_dummies(final[fields], drop_first=True)
        final = final.drop([fields], axis=1, inplace=False)
        if i == 0:
            final_df = df1.copy()
        else:
            final_df = pd.concat([final_df, df1], axis=1)
        i = i + 1
    final_df = pd.concat([final, final_df], axis=1)
    return final_df
final = One_hot_encoding(columns)
final.head()
final.shape
cols = []
count = 1
for column in final.columns:
    cols.append(count)
    count += 1
    continue
final.columns = cols
from sklearn import preprocessing
names = final.columns
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(final)
final = pd.DataFrame(scaled_df, columns=names)
df_train = final.iloc[:1422, :]
df_test = final.iloc[1422:, :]
df_test.shape
X = df_train
X.shape
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
linear_reg = LinearRegression()