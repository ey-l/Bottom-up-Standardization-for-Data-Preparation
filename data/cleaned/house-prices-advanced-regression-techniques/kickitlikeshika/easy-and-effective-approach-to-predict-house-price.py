import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import boxcox1p
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_train.columns
df_test.columns
print(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'])
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.scatter(x=data[var], y=data['SalePrice'])
print(data[:5])

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.scatter(x=data[var], y=data['SalePrice'])
print(data.head())

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
sns.boxplot(x=data[var], y=data['SalePrice'])
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
sns.boxplot(x=data[var], y=data['SalePrice'])
corrmat = df_train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
(f, ax) = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols])
y_train = df_train['SalePrice']
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
test_id = df_test['Id']
all_data = pd.concat([df_train, df_test], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
all_data.drop(missing_data[missing_data['Total'] > 5].index, axis=1, inplace=True)
print(all_data.isnull().sum().max())
print(all_data.info())
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(ntrain)
print(ntest)
print(all_data.shape)
print(all_data.columns)
for col in ('GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data.drop(['Utilities'], axis=1, inplace=True)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)
print(ntrain)
print(ntest)
print(all_data.shape)
print(all_data.columns)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
print(all_data.shape)
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
print(len(numeric_feats))
print(numeric_feats)
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data.head()
all_data = pd.get_dummies(all_data)
all_data.shape
train = all_data[:ntrain]
test = all_data[ntrain:]
print(train.shape)
print(test.shape)
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
scorer = make_scorer(mean_squared_error, greater_is_better=False)

def rmse_CV_train(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse

def rmse_CV_test(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, test, y_test, scoring='neg_mean_squared_error', cv=kf))
    return rmse
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.464, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)