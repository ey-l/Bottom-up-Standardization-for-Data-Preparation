import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(5)
_input1.columns.values
_input1.shape
sns.set()
for i in range(9):
    sns.pairplot(_input1, x_vars=_input1.columns.values[10 * i:10 * (i + 1)], y_vars='SalePrice')
corrmat = _input1.corr()
corrmat.sort_values('SalePrice', ascending=False).index
(f, ax) = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
print(corrmat.sort_values('SalePrice', ascending=False).index)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['TotalBsmtSF'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
data = pd.concat([_input1['SalePrice'], _input1['OverallQual']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['YearBuilt']], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], fit=norm)
ntrain = _input1.shape[0]
ntest = _input0.shape[0]
y_train = _input1.SalePrice.values
all_data = pd.concat((_input1, _input0)).reset_index(drop=True)
x_all = all_data.drop(['SalePrice', 'Id'], axis=1, inplace=True)
print('all_data size is : {}'.format(all_data.shape))
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()
(f, ax) = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
for i in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']:
    all_data[i] = all_data[i].fillna('None')
print(all_data.shape)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
for col in ('MSZoning', 'Functional', 'Electrical', 'GarageCond', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    print(all_data[col].mode()[0])
print(all_data.shape)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
print(all_data.shape)
all_data = all_data.drop(['Utilities'], axis=1)
print(all_data.shape)
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print('\nSkew in numerical features: \n')
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head(10)
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
skewness = skewness[abs(skewness) > 0.75]
print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
for i in skewness.index:
    all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))
print(all_data.shape)
all_data = pd.get_dummies(all_data).reset_index(drop=True)
print(all_data.shape)
X_train = all_data.iloc[:ntrain]
X_test = all_data.iloc[ntrain:]
print(X_train.shape, X_test.shape, y_train.shape)
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
n_folds = 12
kf = KFold(n_splits=12, random_state=42, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X_train):
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
lightgbm = LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.01, n_estimators=7000, max_bin=200, bagging_fraction=0.8, bagging_freq=4, bagging_seed=8, feature_fraction=0.2, feature_fraction_seed=8, min_sum_hessian_in_leaf=11, verbose=-1, random_state=42)
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=6000, max_depth=4, min_child_weight=0, gamma=0.6, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=6e-05, random_state=42)
ridge_alphas = [1e-15, 1e-10, 1e-08, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001, 0.001, 0.05, 0.01, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, ENet, lasso), meta_regressor=xgboost, use_features_in_secondary=True)
scores = {}
score = cv_rmse(svr)
print('SVR: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
score = cv_rmse(ENet)
print('ENet: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['ENet'] = (score.mean(), score.std())
score = cv_rmse(lasso)
print('lasso: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
scores['lasso'] = (score.mean(), score.std())
print('stack_gen')