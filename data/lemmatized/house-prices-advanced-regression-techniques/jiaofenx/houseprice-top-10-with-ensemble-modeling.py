import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1.head()
quantitative = _input1.dtypes[_input1.dtypes != 'object'].index
qualitative = _input1.dtypes[_input1.dtypes == 'object'].index
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], size=2.5)
features = quantitative
standard = _input1[_input1['SalePrice'] < 200000]
pricey = _input1[_input1['SalePrice'] >= 200000]
diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.0).mean() - standard[f].fillna(0.0).mean()) / standard[f].fillna(0.0).mean() for f in features]
sns.barplot(data=diff, x='feature', y='difference')
x = plt.xticks(rotation=90)
saleprice_scaled = StandardScaler().fit_transform(_input1['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
var = 'GrLivArea'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
_input1 = _input1.drop(_input1[(_input1['OverallQual'] < 5) & (_input1['SalePrice'] > 200000)].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4500) & (_input1['SalePrice'] < 300000)].index, inplace=False)
_input1 = _input1.reset_index(drop=True, inplace=False)
ntrain = _input1.shape[0]
ntest = _input0.shape[0]
full_data = pd.concat((_input1, _input0)).reset_index(drop=True)
full_data = full_data.drop(['SalePrice'], axis=1, inplace=False)
total = full_data.isnull().sum().sort_values(ascending=False)
percent = (full_data.isnull().sum() / full_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
full_data = full_data.fillna(full_data.mean(), inplace=False)
full_data = full_data.fillna(full_data.mode().iloc[0], inplace=False)
full_data.isnull().sum().max()
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
full_data = full_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
print('Shape of full_data: {}'.format(full_data.shape))
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
numeric_feats = full_data.dtypes[full_data.dtypes != 'object'].index
skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skew_index = skewed_feats.index
print(skew_index)
for i in skew_index:
    full_data[i] = boxcox1p(full_data[i], boxcox_normmax(full_data[i] + 1))
full_data = pd.get_dummies(full_data)
full_data.head()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, LassoLarsCV, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
X_train = full_data[:ntrain]
X_test = full_data[ntrain:]
y = _input1.SalePrice.values
kf = KFold(n_splits=12, random_state=42, shuffle=True)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y, scoring='neg_mean_squared_error', cv=kf))
    return rmse
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title='Validation')
plt.xlabel('alpha')
plt.ylabel('rmse')
print(cv_ridge.min())