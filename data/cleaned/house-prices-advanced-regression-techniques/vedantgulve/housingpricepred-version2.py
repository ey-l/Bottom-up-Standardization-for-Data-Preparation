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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
id_test = df_test['Id']
print(df_train.shape)
print(df_test.shape)
pd.options.display.max_rows = 40
pd.options.display.max_columns = None
plt.rcParams['figure.figsize'] = (18, 8)
df_train.head()
df_test.head()
df_train.drop(columns='Id', inplace=True)
df_test.drop(columns='Id', inplace=True)
df_train.info()
df_train.describe()
print(df_train.shape)
print(df_test.shape)
sns.heatmap(df_train.isnull(), cmap='Blues', cbar=False, yticklabels=False, xticklabels=df_train.columns)
df_train = df_train.drop(columns=['Alley', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Fence'])
df_test = df_test.drop(columns=['Alley', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Fence'])
feature_cols = [col for col in df_train.columns if col not in ['SalePrice']]
target_col = ['SalePrice']
categorical_cols = [col for col in feature_cols if df_train[col].dtype == 'O']
numeric_cols = [col for col in feature_cols if col not in categorical_cols]
df_train[numeric_cols].isnull().sum()
df_train[categorical_cols].isnull().sum()
df_train_numerical = df_train.select_dtypes(include=['int64', 'float64'])
df_train_numerical.head()
corrmat = df_train_numerical.corr()
sns.heatmap(corrmat, cmap='RdYlGn')
df_train.drop(columns='LotFrontage', inplace=True)
df_test.drop(columns='LotFrontage', inplace=True)
t_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
sns.heatmap(df_train_numerical[t_corr].corr(), annot=True, cmap='RdYlGn')
plt.scatter(df_train.GrLivArea, df_train.SalePrice)
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
df_train = df_train.drop(df_train[df_train['GrLivArea'] > 4000].index)
plt.scatter(df_train.GrLivArea, df_train.SalePrice)
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(df_train.TotalBsmtSF, df_train.SalePrice)
plt.title('TotalBsmtSF vs SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index)
plt.scatter(df_train.TotalBsmtSF, df_train.SalePrice)
plt.title('TotalBsmtSF vs SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.scatter(df_train['1stFlrSF'], df_train.SalePrice)
plt.title('1stFlrSF vs SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('SalePrice')
df_train = df_train.drop(df_train[df_train['1stFlrSF'] > 2750].index)
plt.scatter(df_train['1stFlrSF'], df_train.SalePrice)
plt.title('1stFlrSF vs SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('SalePrice')
plt.scatter(df_train['GarageArea'], df_train.SalePrice)
plt.title('GarageArea vs SalePrice')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
df_train = df_train.drop(df_train[df_train['GarageArea'] > 1200].index)
plt.scatter(df_train['GarageArea'], df_train.SalePrice)
plt.title('GarageArea vs SalePrice')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
print(df_train.shape)
X_train = df_train.loc[:, 'MSSubClass':'SaleCondition']
y_train = df_train['SalePrice']
print(X_train.shape, y_train.shape)
X_test = df_test.loc[:, 'MSSubClass':'SaleCondition']
print(X_test.shape)
df_train.columns
missing_cells = df_train.isnull().sum().sum()
total_cells = np.product(df_train.shape)
percent_missing = missing_cells / total_cells * 100
percent_missing
df_train.loc[:, 'MSSubClass':'Heating'].isnull().sum()
df_train.loc[:, 'HeatingQC':'SalePrice'].isnull().sum()
columns_na_to_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
columns_na_to_mf = ['MasVnrType', 'Electrical']
columns_na_to_avg = ['MasVnrArea', 'GarageYrBlt']
for column in columns_na_to_None:
    imputer1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')