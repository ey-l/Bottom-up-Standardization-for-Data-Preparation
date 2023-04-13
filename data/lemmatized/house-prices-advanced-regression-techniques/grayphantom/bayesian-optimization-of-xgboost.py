import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
print(_input0.shape)
_input1.head()
_input0.head()
test_ID = _input0['Id']
_input1 = _input1.drop(['Id'], axis=1)
_input0 = _input0.drop(['Id'], axis=1)
y_train = _input1['SalePrice']
data = pd.concat([_input1, _input0], ignore_index=True)
data = data.drop(['SalePrice'], axis=1)
data.shape
plt.figure(figsize=(7, 6))
sns.distplot(y_train)
y_train = np.log1p(y_train)
sns.distplot(y_train)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False) * 100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
data['PoolQC'] = data['PoolQC'].fillna('None')
data['MiscFeature'] = data['MiscFeature'].fillna('None')
data['Alley'] = data['Alley'].fillna('None')
data['Fence'] = data['Fence'].fillna('None')
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    data[col] = data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('None')
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data = data.fillna(data.mean())
data.shape
corr = _input1.corr()['SalePrice'].sort_values()[::-1]
c = corr.head(15).index
plt.figure(figsize=(12, 8))
sns.heatmap(_input1[c].corr(), annot=True)
data = data.drop(['GarageCars', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt'], axis=1)
num_f = data.dtypes[data.dtypes != object].index
skew_f = data[num_f].apply(lambda x: skew(x.dropna()))
skew_f = skew_f[skew_f > 0.75]
skew_f = skew_f.index
data[skew_f] = np.log1p(data[skew_f])
data = pd.get_dummies(data)
data.shape
x_train = data[:_input1.shape[0]]
test = data[_input1.shape[0]:]
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
_input1 = xgb.DMatrix(x_train, label=y_train)
_input0 = xgb.DMatrix(test)
def_params = {'min_child_weight': 1, 'max_depth': 6, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 1, 'reg_alpha': 0, 'learning-rate': 0.3, 'silent': 1}
cv_res = xgb.cv(def_params, _input1, nfold=5)
cv_res.tail()
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
params = {'min_child_weight': (0, 50), 'max_depth': (0, 10), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0), 'reg_lambda': (1e-05, 100, 'log-uniform'), 'reg_alpha': (1e-05, 100, 'log-uniform'), 'learning-rate': (0.01, 0.2, 'log-uniform')}
bayes = BayesSearchCV(xgb.XGBRegressor(), params, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)