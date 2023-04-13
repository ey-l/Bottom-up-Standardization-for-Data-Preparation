import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
seed = 50
np.random.seed(seed)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
actual_y = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv').SalePrice
Id = _input0.Id
cols = _input1.columns
cols = cols.drop('SalePrice')
_input1.info()
_input1 = _input1[_input1.SalePrice.notnull()]
cols = cols.drop(['Alley', 'PoolQC', 'MiscFeature'])
_input1.head()
categorical = _input1.select_dtypes(exclude=[np.number])
numerical = _input1.select_dtypes(include=[np.number])
for (idx, feat) in enumerate(numerical.columns.difference(['Price'])):
    ax = sns.jointplot(x=feat, y='SalePrice', data=numerical, kind='scatter')
    ax.set_axis_labels(feat, 'SalePrice')
_input1 = _input1.drop(_input1[_input1['1stFlrSF'] > 4000].index)
_input1['SalePrice'].describe()
g = sns.distplot(_input1['SalePrice'])
for item in g.get_xticklabels():
    item.set_rotation(90)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
for (col, col_data) in _input1.items():
    if is_string_dtype(col_data):
        _input1[col] = _input1[col].astype('category').cat.as_ordered().cat.codes
        _input0[col] = _input0[col].astype('category').cat.as_ordered().cat.codes
y = _input1.SalePrice
X = _input1[cols]
_input0 = _input0[cols]
params_var = {'max_depth': [1, 2, 3, 4, 5], 'gamma': [0, 0.5, 1], 'n_estimators': randint(1, 1001), 'learning_rate': uniform(), 'subsample': uniform(), 'colsample_bytree': uniform()}
params_fixed = {'silent': 1}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
model = RandomizedSearchCV(estimator=XGBRegressor(**params_fixed, seed=seed), param_distributions=params_var, n_iter=10, cv=cv, scoring='r2', random_state=seed)