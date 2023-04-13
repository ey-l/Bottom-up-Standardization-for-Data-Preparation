import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, SGDRegressor, LinearRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
id_test = _input0['Id']
print(_input1.shape)
print(_input0.shape)
pd.options.display.max_rows = 40
pd.options.display.max_columns = None
plt.rcParams['figure.figsize'] = (18, 8)
_input1.head()
_input0.head()
_input1 = _input1.drop(columns='Id', inplace=False)
_input0 = _input0.drop(columns='Id', inplace=False)
_input1.info()
_input1.describe()
print(_input1.shape)
print(_input0.shape)
sns.heatmap(_input1.isnull(), cmap='Blues', cbar=False, yticklabels=False, xticklabels=_input1.columns)
_input1 = _input1.drop(columns=['Alley', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Fence'])
_input0 = _input0.drop(columns=['Alley', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Fence'])
feature_cols = [col for col in _input1.columns if col not in ['SalePrice']]
target_col = ['SalePrice']
categorical_cols = [col for col in feature_cols if _input1[col].dtype == 'O']
numeric_cols = [col for col in feature_cols if col not in categorical_cols]
_input1[numeric_cols].isnull().sum()
_input1[categorical_cols].isnull().sum()
df_train_numerical = _input1.select_dtypes(include=['int64', 'float64'])
df_train_numerical.head()
corrmat = df_train_numerical.corr()
sns.heatmap(corrmat, cmap='RdYlGn')
_input1 = _input1.drop(columns='LotFrontage', inplace=False)
_input0 = _input0.drop(columns='LotFrontage', inplace=False)
t_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
sns.heatmap(df_train_numerical[t_corr].corr(), annot=True, cmap='RdYlGn')
plt.scatter(_input1.GrLivArea, _input1.SalePrice)
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[_input1['GrLivArea'] > 4000].index)
plt.scatter(_input1.GrLivArea, _input1.SalePrice)
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(_input1.TotalBsmtSF, _input1.SalePrice)
plt.title('TotalBsmtSF vs SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[_input1['TotalBsmtSF'] > 3000].index)
plt.scatter(_input1.TotalBsmtSF, _input1.SalePrice)
plt.title('TotalBsmtSF vs SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.scatter(_input1['1stFlrSF'], _input1.SalePrice)
plt.title('1stFlrSF vs SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[_input1['1stFlrSF'] > 2750].index)
plt.scatter(_input1['1stFlrSF'], _input1.SalePrice)
plt.title('1stFlrSF vs SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('SalePrice')
plt.scatter(_input1['GarageArea'], _input1.SalePrice)
plt.title('GarageArea vs SalePrice')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[_input1['GarageArea'] > 1200].index)
plt.scatter(_input1['GarageArea'], _input1.SalePrice)
plt.title('GarageArea vs SalePrice')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
print(_input1.shape)
X_train = _input1.loc[:, 'MSSubClass':'SaleCondition']
y_train = _input1['SalePrice']
print(X_train.shape, y_train.shape)
X_test = _input0.loc[:, 'MSSubClass':'SaleCondition']
print(X_test.shape)
_input1.columns
missing_cells = _input1.isnull().sum().sum()
total_cells = np.product(_input1.shape)
percent_missing = missing_cells / total_cells * 100
percent_missing
_input1.loc[:, 'MSSubClass':'Heating'].isnull().sum()
_input1.loc[:, 'HeatingQC':'SalePrice'].isnull().sum()
columns_na_to_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
columns_na_to_mf = ['MasVnrType', 'Electrical']
columns_na_to_avg = ['MasVnrArea', 'GarageYrBlt']
for column in columns_na_to_None:
    imputer1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')