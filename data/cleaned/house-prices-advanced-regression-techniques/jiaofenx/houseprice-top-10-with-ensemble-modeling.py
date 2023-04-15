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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Skewness: %f' % train_df['SalePrice'].skew())
print('Kurtosis: %f' % train_df['SalePrice'].kurt())
train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop('Id', axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)
train_df.head()
quantitative = train_df.dtypes[train_df.dtypes != 'object'].index
qualitative = train_df.dtypes[train_df.dtypes == 'object'].index
sns.distplot(train_df['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
corrmat = train_df.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size=2.5)
features = quantitative
standard = train_df[train_df['SalePrice'] < 200000]
pricey = train_df[train_df['SalePrice'] >= 200000]
diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.0).mean() - standard[f].fillna(0.0).mean()) / standard[f].fillna(0.0).mean() for f in features]
sns.barplot(data=diff, x='feature', y='difference')
x = plt.xticks(rotation=90)
saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
train_df.drop(train_df[(train_df['OverallQual'] < 5) & (train_df['SalePrice'] > 200000)].index, inplace=True)
train_df.drop(train_df[(train_df['GrLivArea'] > 4500) & (train_df['SalePrice'] < 300000)].index, inplace=True)
train_df.reset_index(drop=True, inplace=True)
ntrain = train_df.shape[0]
ntest = test_df.shape[0]
full_data = pd.concat((train_df, test_df)).reset_index(drop=True)
full_data.drop(['SalePrice'], axis=1, inplace=True)
total = full_data.isnull().sum().sort_values(ascending=False)
percent = (full_data.isnull().sum() / full_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
full_data.fillna(full_data.mean(), inplace=True)
full_data.fillna(full_data.mode().iloc[0], inplace=True)
full_data.isnull().sum().max()
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
full_data = full_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)
print('Shape of full_data: {}'.format(full_data.shape))
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
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
y = train_df.SalePrice.values
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