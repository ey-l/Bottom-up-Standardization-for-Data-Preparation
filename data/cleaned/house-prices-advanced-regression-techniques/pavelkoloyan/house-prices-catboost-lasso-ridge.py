import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train['sample'] = 1
test['sample'] = 0
df = test.append(train, sort=False).reset_index(drop=True)
test_id = test['Id']
plt.figure(figsize=(15, 15))
(fig, ax) = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(df['SalePrice'], ax=ax[0, 0])
sns.distplot(np.log(df['SalePrice']), ax=ax[0, 1])
stats.probplot(df['SalePrice'], plot=ax[1, 0], dist='norm')
stats.probplot(np.log(df['SalePrice']), plot=ax[1, 1], dist='norm')
fig.show()
print('========================================')
print('Skew SalePrice: ', df['SalePrice'].skew())
print('Skew log SalePrice: ', np.log(df['SalePrice']).skew())
print('========================================')
df.describe().T
df.describe(include='O').T
num = df.select_dtypes(include=[np.number])
hight_corr = num.corr()['SalePrice'][num.corr()['SalePrice'] > 0.1].sort_values().index
plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(num[hight_corr].corr(), linecolor='white', annot=True, linewidths=1, vmax=0.5, vmin=-0.5)
num.skew().sort_values()
cat = df.select_dtypes(exclude=[np.number])
li_cat_feats = list(cat)
nr_rows = 15
nr_cols = 3
(fig, axs) = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols * 7, nr_rows * 3))
for r in range(0, nr_rows):
    for c in range(0, nr_cols):
        i = r * nr_cols + c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=df['SalePrice'], data=df, ax=axs[r][c])
plt.tight_layout()

df_nan = df.loc[:, df.isna().any().values]
gradient = ['#2f4c28', '#3a6049', '#4e7f76', '#8f6e60', '#a96d46', '#c3762c', '#b14404', '#a03031']
msno.bar(df_nan, color=gradient, figsize=(30, 3), sort='ascending')

def preproc_nan(df):
    df['FireplaceQu'].fillna('NA', inplace=True)
    df['Fence'].fillna('NA', inplace=True)
    df['Alley'].fillna('NA', inplace=True)
    df['PoolQC'].fillna('NA', inplace=True)
    df['GarageType'].fillna('NA', inplace=True)
    df['GarageFinish'].fillna('NA', inplace=True)
    df['GarageQual'].fillna('NA', inplace=True)
    df['GarageCond'].fillna('NA', inplace=True)
    df['BsmtFinType1'].fillna('NA', inplace=True)
    df['BsmtFinType2'].fillna('NA', inplace=True)
    df['BsmtExposure'].fillna('NA', inplace=True)
    df['BsmtCond'].fillna('NA', inplace=True)
    df['BsmtQual'].fillna('NA', inplace=True)
    df['MSZoning'] = df['MSZoning'].fillna('RL')
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
    df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['Electrical'] = df['Electrical'].fillna('SBrkr')
    df['KitchenQual'] = df['KitchenQual'].fillna('TA')
    df['Functional'] = df['Functional'].fillna('Typ')
    df['SaleType'] = df['SaleType'].fillna('WD')
    return df

def impute_knn(df):
    ttn = df.select_dtypes(include=[np.number])
    ttc = df.select_dtypes(exclude=[np.number])
    cols_nan = ttn.columns[ttn.isna().any()].tolist()
    cols_no_nan = ttn.columns.difference(cols_nan).values
    for col in cols_nan:
        imp_test = ttn[ttn[col].isna()]
        imp_train = ttn.dropna()
        model = KNeighborsRegressor(n_neighbors=5)