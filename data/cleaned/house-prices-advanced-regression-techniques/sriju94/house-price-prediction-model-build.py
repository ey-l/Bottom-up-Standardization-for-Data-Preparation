import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import GridSearchCV
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
train_df['SalePrice'].describe()
sns.distplot(train_df['SalePrice'])
print(train_df['SalePrice'].skew())
print(train_df['SalePrice'].kurt())
corrmatrix = train_df.corr().abs()
sns.heatmap(corrmatrix)
corrmatrix[corrmatrix['SalePrice'] < 0.3].index
corr_var_list = ['Id', 'MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
train_df.shape
data = pd.concat([train_df, test_df])
print(data.shape)
data.head()
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
null_feature = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
data.drop(null_feature, axis=1, inplace=True)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
data.info()
cols_none = ['MSZoning', 'BsmtCond', 'Utilities', 'BsmtExposure', 'Exterior1st', 'Exterior2nd', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Electrical', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'KitchenQual', 'SaleType', 'MasVnrType', 'Functional']
for col in cols_none:
    data[col] = data[col].fillna('None')
data['LotFrontage'] = data.groupby('MSZoning').LotFrontage.transform(lambda row: row.fillna(row.mean()))
num_col = data.dtypes[data.dtypes != 'object'].index.to_list()
cat_col = data.dtypes[data.dtypes == 'object'].index.to_list()
ordinal_encoder = OrdinalEncoder()
data[cat_col] = ordinal_encoder.fit_transform(data[cat_col])
my_imputer = SimpleImputer()
imputed_data = pd.DataFrame(my_imputer.fit_transform(data))
imputed_data.columns = data.columns
imputed_data.head()
imputed_data['Remodeling'] = np.where(imputed_data['YearBuilt'] == imputed_data['YearRemodAdd'], 0, 1)
imputed_data['RemodGap'] = imputed_data['YearRemodAdd'] - imputed_data['YearBuilt']
imputed_data['HouseAge'] = 2021 - imputed_data['YearBuilt']
imputed_data['TotBath'] = imputed_data['BsmtFullBath'] + 0.5 * imputed_data['BsmtHalfBath'] + imputed_data['FullBath'] + 0.5 * imputed_data['HalfBath']
imputed_data['CarpetArea'] = imputed_data['TotalBsmtSF'] + imputed_data['1stFlrSF'] + imputed_data['2ndFlrSF']
imputed_data['OutSideArea'] = imputed_data.WoodDeckSF + imputed_data.OpenPorchSF + imputed_data.EnclosedPorch + imputed_data['3SsnPorch'] + imputed_data.ScreenPorch + imputed_data.PoolArea
feature = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
X_cluster = pd.DataFrame(imputed_data[feature])
objective_function = []
for i in range(1, 6):
    clustering = KMeans(n_clusters=i, init='k-means++')