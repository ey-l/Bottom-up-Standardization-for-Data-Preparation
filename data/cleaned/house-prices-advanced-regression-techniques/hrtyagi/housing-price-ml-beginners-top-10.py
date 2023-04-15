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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train_Id = train['Id']
test_Id = test['Id']
train.describe()
test.head()
test.describe()
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
train.info()
test.info()
train.isnull().sum().sort_values(ascending=False)[0:20]
test.isnull().sum().sort_values(ascending=False)[0:35]
list_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt']
for col in list_drop:
    del train[col]
    del test[col]
train.isnull().sum().sort_values(ascending=False)[0:15]
test.isnull().sum().sort_values(ascending=False)[0:30]
train.LotFrontage.value_counts(dropna=False)
train.LotFrontage.fillna(train.LotFrontage.mean(), inplace=True)
test.LotFrontage.fillna(test.LotFrontage.mean(), inplace=True)
print(train.BsmtCond.value_counts(dropna=False))
print(test.BsmtCond.value_counts(dropna=False))
list_fill_train = ['BsmtCond', 'BsmtQual', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'BsmtFinType2', 'BsmtExposure', 'FireplaceQu', 'MasVnrArea']
for j in list_fill_train:
    train[j] = train[j].fillna(train[j].mode()[0])
    test[j] = test[j].fillna(train[j].mode()[0])
print(train.isnull().sum().sort_values(ascending=False)[0:5])
print(test.isnull().sum().sort_values(ascending=False)[0:20])
train.dropna(inplace=True)
train.shape
list_test_str = ['BsmtFinType1', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'MSZoning']
list_test_num = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']
for item in list_test_str:
    test[item] = test[item].fillna(test[item].mode()[0])
for item in list_test_num:
    test[item] = test[item].fillna(test[item].mean())
print(train.isnull().sum().sort_values(ascending=False)[0:5])
print(test.isnull().sum().sort_values(ascending=False)[0:5])
test.shape
Y = train['SalePrice']
del train['Id']
del test['Id']
del train['SalePrice']
test.shape
train.shape
final = pd.concat([train, test], axis=0)
final.shape
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

def One_hot_encoding(columns):
    final_df = final
    i = 0
    for fields in columns:
        df1 = pd.get_dummies(final[fields], drop_first=True)
        final.drop([fields], axis=1, inplace=True)
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