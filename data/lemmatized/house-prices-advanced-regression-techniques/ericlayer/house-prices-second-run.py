import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.ensemble import IsolationForest, RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import shap
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_ind1 = _input1.index
test_ind1 = _input0.index
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
house_all = pd.concat([_input1, _input0], axis=0, sort=False)
y = house_all['SalePrice']
house_all.head()
_input1 = _input1.drop('Id', axis=1)
_input0 = _input0.drop('Id', axis=1)
house_all = house_all.drop('Id', axis=1)
house_all['MSSubClass'] = house_all['MSSubClass'].astype(str)
_input1.dtypes.unique()
int64_cols = [col for col in house_all.columns if house_all[col].dtype == 'int64']
float64_cols = [col for col in house_all.columns if house_all[col].dtype == 'float64']
object_cols = [col for col in house_all.columns if house_all[col].dtype == 'O']
i = 1
plt.figure(figsize=(20, 40))
for col in int64_cols:
    plt.subplot(11, 4, i)
    sns.histplot(x=col, data=house_all.reset_index(), kde=True)
    i = i + 1
i = 1
plt.figure(figsize=(20, 40))
for col in int64_cols:
    plt.subplot(11, 4, i)
    sns.scatterplot(x=col, y='SalePrice', data=house_all.iloc[:1460])
    i = i + 1
i = 1
plt.figure(figsize=(20, 40))
for col in float64_cols:
    plt.subplot(11, 4, i)
    sns.histplot(x=col, data=house_all.reset_index(), kde=True)
    i = i + 1
i = 1
plt.figure(figsize=(20, 40))
for col in float64_cols:
    plt.subplot(11, 4, i)
    sns.scatterplot(x=col, y='SalePrice', data=house_all.iloc[:1460])
    i = i + 1
i = 1
plt.figure(figsize=(20, 40))
for col in object_cols:
    plt.subplot(11, 4, i)
    sns.histplot(x=col, data=house_all.reset_index(), kde=True)
    i = i + 1
i = 1
plt.figure(figsize=(20, 40))
for col in object_cols:
    plt.subplot(11, 4, i)
    sns.scatterplot(x=col, y='SalePrice', data=house_all.iloc[:1460])
    i = i + 1
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(x='SalePrice', data=house_all.loc[train_ind1], kde=True, ax=ax[0])
sns.boxplot(x=_input1['SalePrice'], ax=ax[1])
fig.suptitle('Sale Price Distribution of Training Dataset')
fig.show()
num_cat_vars = ['BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'MoSold', 'YrSold']
num_cont_vars = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(house_all.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True)
plt.figure(figsize=(40, 20))
mask = np.triu(np.ones_like(house_all.loc[:, :'SaleCondition'].corr(), dtype=np.bool))
heatmap = sns.heatmap(house_all.loc[:, :'SaleCondition'].corr(), annot=True, mask=mask)
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
house_all = house_all.reset_index(drop=True, inplace=False)
missing_vals = pd.DataFrame(house_all.isnull().sum().sort_values(ascending=False)[:25]).reset_index()
missing_vals.columns = ['Features', 'Missing Values']
fig = plt.figure(figsize=(12, 8))
g = sns.barplot(x='Features', y='Missing Values', data=missing_vals, order=missing_vals.sort_values('Missing Values', ascending=False).Features)
g.set_title('Missing values by Feature')
plt.xticks(rotation=90)
missing_vals
house_all = house_all.drop(['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal', 'Alley', 'Fence'], axis=1, inplace=False)
house_all[house_all.loc[:, ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 'GarageType']].isnull().all(axis=1)].shape
house_all['GarageYrBlt'] = house_all['GarageYrBlt'].fillna(house_all['YearBuilt'], inplace=False)
house_all['GarageYrBlt'] = house_all['GarageYrBlt'].astype(int)
house_all['GarageCond'] = house_all['GarageCond'].fillna('No', inplace=False)
house_all['GarageQual'] = house_all['GarageQual'].fillna('No', inplace=False)
house_all['GarageFinish'] = house_all['GarageFinish'].fillna('No', inplace=False)
house_all['GarageType'] = house_all['GarageType'].fillna('No', inplace=False)
house_all['FireplaceQu'] = house_all['FireplaceQu'].fillna('No', inplace=False)
bsmt_fill_inds = house_all[house_all.loc[:, ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().all(axis=1)].index
house_all['BsmtQual'] = house_all['BsmtQual'].fillna('No', inplace=False)
house_all['BsmtCond'] = house_all['BsmtCond'].fillna('No', inplace=False)
house_all['BsmtExposure'] = house_all['BsmtExposure'].fillna('No', inplace=False)
house_all['BsmtFinType1'] = house_all['BsmtFinType1'].fillna('No', inplace=False)
house_all['BsmtFinType2'] = house_all['BsmtFinType2'].fillna('No', inplace=False)
house_all['GarageCars'] = house_all['GarageCars'].fillna(0, inplace=False)
house_all['GarageArea'] = house_all['GarageArea'].fillna(0, inplace=False)
knn_cols = ['MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
freq_cols = ['MasVnrType', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'SaleType']
for col in knn_cols:
    imp_knn = KNNImputer(n_neighbors=5)
    temp = imp_knn.fit_transform(house_all[[col]]).ravel()
    house_all[col] = temp
for col in freq_cols:
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    temp = imp_freq.fit_transform(house_all[[col]]).ravel()
    house_all[col] = temp
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(house_all.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True)
int64_cols = [col for col in house_all.columns if house_all[col].dtype == 'int64']
float64_cols = [col for col in house_all.columns if house_all[col].dtype == 'float64']
object_cols = [col for col in house_all.columns if house_all[col].dtype == 'O']
house_trainv2 = house_all.loc[:1459]
house_trainv2.head()
i = 1
plt.figure(figsize=(22, 40))
float64_outlier_indices = []
for col in float64_cols:
    plt.subplot(10, 4, i)
    plt.subplots_adjust(wspace=0.3)
    x1 = house_trainv2[col].values.reshape(-1, 1)
    x2 = house_trainv2['SalePrice'].values.reshape(-1, 1)
    x = np.concatenate((x1, x2), axis=1)
    clf = IForest(contamination=0.005, random_state=3)