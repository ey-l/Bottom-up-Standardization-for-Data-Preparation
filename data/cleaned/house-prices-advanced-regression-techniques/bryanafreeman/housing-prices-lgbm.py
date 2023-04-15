import pdb
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn_pandas import DataFrameMapper
from operator import itemgetter
import lightgbm as lgb
import os

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
train_csv_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_csv_path = '_data/input/house-prices-advanced-regression-techniques/test.csv'
train_set = pd.read_csv(train_csv_path)
test_set = pd.read_csv(test_csv_path)
train_data = train_set.copy()
test_data = test_set.copy()
train_ids = train_data['Id'].copy()
test_ids = test_data['Id'].copy()
print('Test data original columns: {}'.format(train_data.columns.to_list()))
print('Train data original shape: {}'.format(train_data.shape))
print('Test data original shape: {}'.format(test_data.shape))
train_data.head(5)
train_data['SalePrice'].describe()
plt.figure(figsize=(16, 6))
sns.distplot(train_data['SalePrice'])
(fig, ax) = plt.subplots(figsize=(23, 10))
ax.set(yscale='log')
sns.barplot(x='Neighborhood', y='SalePrice', data=train_data, estimator=np.mean)

var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', figsize=(16, 6), ylim=(0, 800000))
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
(f, ax) = plt.subplots(figsize=(22, 12))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
corrs_matrix = train_data.corr()
(f, ax) = plt.subplots(figsize=(12, 12))
sns.heatmap(corrs_matrix, vmax=0.8, square=True)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size=2.5)

total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum() / train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('Low range of distribution:')
print(low_range)
print('\nHigh range of the distribution:')
print(high_range)
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), figsize=(16, 8))
train_data.sort_values(by='GrLivArea', ascending=False)[:2]
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
sns.distplot(train_data['SalePrice'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
train_data['SalePrice'] = np.log(train_data['SalePrice'])
sns.distplot(train_data['SalePrice'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
sns.distplot(train_data['GrLivArea'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
sns.distplot(train_data['GrLivArea'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)
sns.distplot(train_data['TotalBsmtSF'], fit=stats.norm)
fig = plt.figure()
res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
train_data = train_set.copy()
train_index = train_data.shape[0]
test_index = test_data.shape[0]
target = train_data.SalePrice.values
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('FireplaceQu')
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')
for col in columns:
    encoder = LabelEncoder()