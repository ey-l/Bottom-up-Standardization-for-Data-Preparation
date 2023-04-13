import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head(10)
_input1[_input1.columns[_input1.isna().sum() > 0]].isna().mean() * 100
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], inplace=False, axis=1)
_input1[_input1.columns[_input1.isna().sum() > 0]].isna().mean() * 100
x = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
for i in x.columns:
    if x[i].dtype != 'object':
        sns.boxplot(x[i])
        plt.title(i)
for i in x.columns:
    if x[i].dtype != 'object':
        value_z = (x[i] - x[i].mean()) / x[i].std()
        sns.distplot(value_z)
x['MSSubClass'][x['MSSubClass'] > 170] = 170
x['LotFrontage'][x['LotFrontage'] > 190] = 190
x['LotArea'][x['LotArea'] > 60000] = 60000
x['OverallCond'][x['OverallCond'] > 8] = 8
x['YearBuilt'][x['YearBuilt'] < 1879] = 1879
x['MasVnrArea'][x['MasVnrArea'] > 1050] = 1050
x['BsmtFinSF1'][x['BsmtFinSF1'] > 3000] = 3000
x['BsmtUnfSF'][x['BsmtUnfSF'] > 2200] = 2200
x['TotalBsmtSF'][x['TotalBsmtSF'] > 4000] = 4000
x['1stFlrSF'][x['1stFlrSF'] > 3000] = 3000
x['GrLivArea'][x['GrLivArea'] > 4100] = 4100
x['BsmtFullBath'][x['BsmtFullBath'] > 2.5] = 2.5
x['GarageArea'][x['GarageArea'] > 1300] = 1300
x['WoodDeckSF'][x['WoodDeckSF'] > 650] = 650
x['OpenPorchSF'][x['OpenPorchSF'] > 400] = 400
x[x.columns[x.isna().sum() > 0]]
x[x.columns[x.isna().sum() > 0]].hist(figsize=(20, 20))
numerical = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for i in numerical:
    x[i] = x[i].fillna(x[i].median())
categorical = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for i in categorical:
    x[i] = x[i].fillna(x[i].mode()[0])
x[x.columns[x.isna().sum() > 0]].isna().mean() * 100
x_en = pd.get_dummies(x, drop_first=True)
x_en.head()
mm_scaler = MinMaxScaler()
x_scaled = pd.DataFrame(mm_scaler.fit_transform(x_en), columns=x_en.columns)
x_scaled.head()
target_scaler = MinMaxScaler()
y_data = pd.DataFrame(y)