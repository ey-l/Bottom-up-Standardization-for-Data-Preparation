import numpy as np
import pandas as pd
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = train_data
test = test_data
train_test_data = pd.concat([train, test], axis=0, ignore_index=True)
train_test_data.describe()
missing_ratio = train_test_data.isnull().sum() / len(train_test_data)
missing_ratio = missing_ratio[missing_ratio.values > 0]
missing_ratio_low = missing_ratio[missing_ratio.values <= 0.1]
missing_ratio_high = missing_ratio[missing_ratio.values > 0.1]
fig = plt.figure(figsize=(10, 4), dpi=100)
plt.subplot(1, 2, 1)
sns.barplot(x=missing_ratio_low.index, y=missing_ratio_low.values)
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
sns.barplot(x=missing_ratio_high.index, y=missing_ratio_high.values)
plt.xticks(rotation=90)
fig = plt.figure(figsize=(6, 4), dpi=100)
sns.heatmap(train_test_data.corr(), cmap='viridis')
plt.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({'price': train['SalePrice'], 'log(price + 1)': np.log1p(train['SalePrice'])})
prices.hist()
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
train['MiscFeature'] = train['MiscFeature'].replace(to_replace=['Gar2', 'Othr', 'TenC'], value='Other')
train['MiscFeature'] = train['MiscFeature'].fillna(value='Other')
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
test['MiscFeature'] = test['MiscFeature'].replace(to_replace=['Gar2', 'Othr', 'TenC'], value='Other')
test['MiscFeature'] = test['MiscFeature'].fillna(value='Other')
train = train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence'], axis=1)
test = test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence'], axis=1)
train['MSZoning'] = train['MSZoning'].fillna(value='RL')
train['Exterior1st'] = train['Exterior1st'].fillna(value='VinylSd')
train['Exterior2nd'] = train['Exterior2nd'].fillna(value='VinylSd')
train['MasVnrType'] = train['MasVnrType'].fillna(value='None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(value=0)
train[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = train[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna(value='None')
train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = train[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].fillna(value=0)
train['Electrical'] = train['Electrical'].fillna(value='SBrkr')
train['KitchenQual'] = train['KitchenQual'].fillna(value='TA')
train['Functional'] = train['Functional'].fillna(value='Typ')
train[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = train[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna(value='None')
train[['GarageCars', 'GarageArea']] = train[['GarageCars', 'GarageArea']].fillna(value=0)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(value=1900)
train['SaleType'] = train['SaleType'].fillna(value='WD')
test['MSZoning'] = test['MSZoning'].fillna(value='RL')
test['Exterior1st'] = test['Exterior1st'].fillna(value='VinylSd')
test['Exterior2nd'] = test['Exterior2nd'].fillna(value='VinylSd')
test['MasVnrType'] = test['MasVnrType'].fillna(value='None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(value=0)
test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna(value='None')
test[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = test[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].fillna(value=0)
test['Electrical'] = test['Electrical'].fillna(value='SBrkr')
test['KitchenQual'] = test['KitchenQual'].fillna(value='TA')
test['Functional'] = test['Functional'].fillna(value='Typ')
test[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = test[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna(value='None')
test[['GarageCars', 'GarageArea']] = test[['GarageCars', 'GarageArea']].fillna(value=0)
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(value=1900)
test['SaleType'] = test['SaleType'].fillna(value='WD')
train = train.drop(['Utilities', 'Id'], axis=1)
test = test.drop(['Utilities', 'Id'], axis=1)
train['SalePrice'] = np.log1p(train['SalePrice'])
numeric_feats = test.dtypes[test.dtypes != 'object'].index
skewed_feats = test[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
lam = 0.15
for feat in skewed_feats:
    train[feat] = boxcox1p(train[feat], lam)
    test[feat] = boxcox1p(test[feat], lam)
y = train['SalePrice']
train = train.drop('SalePrice', axis=1)
train_test_data = pd.concat([train, test], axis=0, ignore_index=True)
train_test_data = pd.get_dummies(train_test_data, drop_first=True)
X = train_test_data.iloc[:1460]
X_submission = train_test_data.iloc[1460:]
kf = KFold(n_splits=12, random_state=42, shuffle=True)
lightgbm = LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.01, n_estimators=7000, max_bin=200, bagging_fraction=0.8, bagging_freq=4, bagging_seed=8, feature_fraction=0.2, feature_fraction_seed=8, min_sum_hessian_in_leaf=11, verbose=-1, random_state=42)
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=4, min_child_weight=0, gamma=0.6, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=6e-05, random_state=42)
ridge_alphas = [1e-15, 1e-10, 1e-08, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001, 0.001, 0.05, 0.01, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
gbr = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
rf = RandomForestRegressor(n_estimators=1200, max_depth=15, min_samples_split=5, min_samples_leaf=5, max_features=None, oob_score=True, random_state=42)
stack = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf), meta_regressor=xgboost, use_features_in_secondary=True)