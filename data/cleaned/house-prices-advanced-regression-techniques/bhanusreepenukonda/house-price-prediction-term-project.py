import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import seaborn as sb
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.head()
train[train.SalePrice > 300000]
train.tail()
train.info()
train.describe()
correlation_train = train.corr()
sb.set(font_scale=2)
plt.figure(figsize=(50, 35))
ax = sb.heatmap(correlation_train, annot=True, annot_kws={'size': 25}, fmt='.1f', cmap='PiYG', linewidths=0.5)
correlation_train.columns
corr_dict = correlation_train['SalePrice'].sort_values(ascending=False).to_dict()
important_columns = []
for (key, value) in corr_dict.items():
    if (value > 0.1) & (value < 0.8) | (value <= -0.1):
        important_columns.append(key)
important_columns
plt.figure(figsize=(40, 20))
sb.set(font_scale=1.5)
sb.boxplot(x='YearBuilt', y='SalePrice', data=train)
sb.swarmplot(x='YearBuilt', y='SalePrice', data=train, color='.25')
plt.xticks(weight='bold', rotation=90)
test.head()
test.tail()
test.info()
test.describe()
train_test = pd.concat([train, test], axis=0, sort=False)
train_test.head()
pd.set_option('display.max_rows', 5000)
train_test_null_info = pd.DataFrame(train_test.isnull().sum(), columns=['Count of NaN'])
train_test_dtype_info = pd.DataFrame(train_test.dtypes, columns=['DataTypes'])
train_tes_info = pd.concat([train_test_null_info, train_test_dtype_info], axis=1)
train_tes_info
train_test.loc[train_test['Fireplaces'] == 0, 'FireplaceQu'] = 'Nothing'
train_test['LotFrontage'] = train_test['LotFrontage'].fillna(train_test.groupby('1stFlrSF')['LotFrontage'].transform('mean'))
train_test['LotFrontage'].interpolate(method='linear', inplace=True)
train_test['LotFrontage'] = train_test['LotFrontage'].astype(int)
train_test['MasVnrArea'] = train_test['MasVnrArea'].fillna(train_test.groupby('MasVnrType')['MasVnrArea'].transform('mean'))
train_test['MasVnrArea'].interpolate(method='linear', inplace=True)
train_test['MasVnrArea'] = train_test['MasVnrArea'].astype(int)
train_test['Fence'] = train_test['Fence'].fillna('None')
train_test['FireplaceQu'] = train_test['FireplaceQu'].fillna('None')
train_test['Alley'] = train_test['Alley'].fillna('None')
train_test['PoolQC'] = train_test['PoolQC'].fillna('None')
train_test['MiscFeature'] = train_test['MiscFeature'].fillna('None')
train_test.loc[train_test['BsmtFinSF1'] == 0, 'BsmtFinType1'] = 'Unf'
train_test.loc[train_test['BsmtFinSF2'] == 0, 'BsmtQual'] = 'TA'
train_test['YrBltRmd'] = train_test['YearBuilt'] + train_test['YearRemodAdd']
train_test['Total_Square_Feet'] = train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'] + train_test['TotalBsmtSF']
train_test['Total_Bath'] = train_test['FullBath'] + 0.5 * train_test['HalfBath'] + train_test['BsmtFullBath'] + 0.5 * train_test['BsmtHalfBath']
train_test['Total_Porch_Area'] = train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF']
train_test['exists_pool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_garage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_fireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_bsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['old_house'] = train_test['YearBuilt'].apply(lambda x: 1 if x < 1990 else 0)
for i in train_test.columns:
    if 'SalePrice' not in i:
        if 'object' in str(train_test[str(i)].dtype):
            train_test[str(i)] = train_test[str(i)].fillna(method='ffill')
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'YrSold', 'MoSold', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond')
for col in columns:
    lbl_enc = LabelEncoder()