import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head(6)
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test.head(6)
train.info()
train.describe().transpose()
for i in range(train.shape[1]):
    print(train.columns[i], '-', train.iloc[:, i].isnull().sum())
for i in range(test.shape[1]):
    print(test.columns[i], '-', test.iloc[:, i].isnull().sum())
fig = plt.figure(figsize=(15, 8))
sns.distplot(train['SalePrice'], bins=26, color='brown')
sns.set_style('white')
sns.set_context('poster', font_scale=2)
plt.tight_layout()
print('Skewness: ' + str(train['SalePrice'].skew()))
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(30, 20))
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask=mask, annot=True, center=0, cbar='coolwarm')
plt.tight_layout()
train.corr()['SalePrice'].sort_values(ascending=False)[1:]
fig = plt.figure(figsize=(15, 8))
sns.scatterplot(x='OverallQual', y='SalePrice', data=train)
sns.set_style('whitegrid')
sns.set_context('poster', font_scale=2)
plt.tight_layout()
fig = plt.figure(figsize=(15, 8))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.5)
plt.tight_layout()
fig = plt.figure(figsize=(15, 8))
sns.scatterplot(x='GarageArea', y='SalePrice', data=train)
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=2)
plt.tight_layout()
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
previous_train = train.copy()
print(train.shape)
train['SalePrice'] = np.log1p(train['SalePrice'])
train.drop(columns=['Id'], axis=1, inplace=True)
test.drop(columns=['Id'], axis=1, inplace=True)
y = train['SalePrice'].reset_index(drop=True)
previous_train = train.copy()
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
for i in range(all_data.shape[1]):
    print(all_data.columns[i], '-', all_data.iloc[:, i].isnull().sum())
missing_val_col = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None')
missing_val_col2 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']
for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
sum = 0
for i in range(all_data.shape[1]):
    sum = sum + all_data.iloc[:, i].isnull().sum()
print(sum)
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_feats
fig = plt.figure(figsize=(15, 8))
sns.distplot(train['1stFlrSF'], bins=26, color='brown')
sns.set_style('white')
sns.set_context('poster', font_scale=2)
plt.tight_layout()

def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """
    from scipy.stats import skew
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax
    numeric_feats = df.dtypes[df.dtypes != 'object'].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
fixing_skewness(all_data)
sns.distplot(all_data['1stFlrSF'])
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
all_data['Total_sqr_footage'] = all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['Total_Bathrooms'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['Total_porch_sf'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data.shape
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.shape
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit
overfitted_features = overfit_reducer(X)
X = X.drop(overfitted_features, axis=1)
X_sub = X_sub.drop(overfitted_features, axis=1)
(X.shape, y.shape, X_sub.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
alpha_ridge = [-3, -2, -1, 1e-15, 1e-10, 1e-08, 1e-05, 0.0001, 0.001, 0.01, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ridge = Ridge(alpha=i, normalize=True)