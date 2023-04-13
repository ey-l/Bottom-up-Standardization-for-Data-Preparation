import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import plotly.express as px
import plotly.graph_objects as go
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.info()
_input0.head()
_input1.select_dtypes(np.number).hist(bins=50, figsize=(30, 20), color='orange')
corr_matrix = _input1.select_dtypes(np.number).corr()
corr = corr_matrix['SalePrice'].sort_values(ascending=False)
print(corr)
indexNames = corr[abs(corr.values) < 0.4].index.values
indexNames = np.setdiff1d(indexNames, ['Id', 'MSSubClass'])
mask = np.triu(np.ones_like(corr, dtype=np.bool))
corr_matrix = corr_matrix.mask(mask)
fig = px.imshow(corr_matrix, text_auto=True)
fig.layout.height = 1000
fig.layout.width = 1000
fig.show()
y_target = _input1['SalePrice']
test_ids = _input0.Id
train_v0 = _input1.drop(['Id', 'SalePrice'], axis=1)
test_v0 = _input0.drop('Id', axis=1)
data_v0 = pd.concat([train_v0, test_v0], axis=0)
data_v0
data_v0.info()
y_target
data_v0['MSSubClass'] = data_v0['MSSubClass'].astype(str)
data_v0.MSSubClass.dtype
cat_mode_cols = ['MasVnrType', 'MSZoning', 'Functional', 'Utilities', 'Exterior2nd', 'KitchenQual', 'Electrical', 'Exterior1st', 'SaleType']
for col in cat_mode_cols:
    data_v0[col] = data_v0[col].fillna(data_v0[col].mode()[0], inplace=False)
cat_None_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 'PoolQC', 'Fence', 'MiscFeature']
for col in cat_None_cols:
    data_v0[col] = data_v0[col].fillna('None', inplace=False)
ordinal_cols = {'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Electrical': {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 'CentralAir': {'N': 0, 'Y': 1}, 'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}}
data_v0 = data_v0.replace(ordinal_cols, inplace=False)

def display_missing(train, cols):
    mis_val = _input1.isna().sum().sort_values(ascending=False)
    mis_val_per = (mis_val / len(_input1) * 100).sort_values(ascending=False).round(1)
    mis_val_table = pd.concat([mis_val, mis_val_per], axis=1, keys=['# Missing Values', '% Total Missing'])
    return mis_val_table.head(cols)
display_missing(data_v0.select_dtypes('object'), 22)
data_v1 = data_v0.copy()
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(data_v1['LotArea'], kde=True, fit=norm)
plt.title('Without Log Transform')
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(data_v1['LotArea']), kde=True, fit=norm, color='darkblue')
plt.xlabel('Log LotArea ')
plt.title('With Log Transform')
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(data_v1['GrLivArea'], kde=True, fit=norm)
plt.title('Without Log Transform')
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(data_v1['GrLivArea']), kde=True, fit=norm, color='darkblue')
plt.xlabel('Log GrLivArea ')
plt.title('With Log Transform')
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(data_v1['TotalBsmtSF'], kde=True, fit=norm)
plt.title('Without Log Transform')
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(data_v1['TotalBsmtSF']), kde=True, fit=norm, color='darkblue')
plt.xlabel('Log TotalBsmtSF ')
plt.title('With Log Transform')

def display_skew_kurt(df, cols):
    skew = df.select_dtypes(np.number).skew()
    abs_skew = abs(skew)
    kurt = df.select_dtypes(np.number).kurt()
    skew_kurt_table = pd.concat([skew, abs_skew, kurt], axis=1, keys=['Skew', 'Absolute Skew', 'Kurtosis']).sort_values('Skew', ascending=False)
    skew_kurt_table['Skewed'] = skew_kurt_table['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)
    return skew_kurt_table
skew_kurt_df = display_skew_kurt(data_v1, 20)
for col in skew_kurt_df.query('Skewed == True').index.values:
    data_v1[col] = np.log1p(data_v1[col])
data_v1['MoSold'] = -np.cos(5.236) * data_v1['MoSold']
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(y_target, kde=True, fit=norm)
plt.title('Without Log Transform')
plt.subplot(1, 2, 2)
sns.distplot(np.log(y_target), kde=True, fit=norm, color='darkblue')
plt.xlabel('Log SalePrice')
plt.title('With Log Transform')
y_target_log = np.log(y_target)
data_v2 = data_v1.copy()
data_v2 = pd.get_dummies(data_v2)
data_v2
data_v3 = data_v2.copy()
scaler = MinMaxScaler()
data_v3 = pd.DataFrame(scaler.fit_transform(data_v2), columns=data_v3.columns)
data_v3.head()
imputer = KNNImputer(n_neighbors=5)
data_v3 = pd.DataFrame(imputer.fit_transform(data_v3), columns=data_v3.columns)
data_v3.isna().any()
data_v3.isna().sum()
data_v4 = data_v3.copy()
train_final = data_v3.loc[:train_v0.index.max(), :].copy()
test_final = data_v3.loc[train_v0.index.max() + 1:, :].reset_index(drop=True).copy()
(X_train, X_val, y_train, y_val) = train_test_split(train_final, y_target, train_size=0.8, test_size=0.2, random_state=0)
forest_model = RandomForestRegressor(n_estimators=500, max_depth=10)