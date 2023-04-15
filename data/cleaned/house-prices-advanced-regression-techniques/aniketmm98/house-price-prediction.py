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

data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
actual_y = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv').SalePrice
Id = test.Id
cols = data.columns
cols = cols.drop('SalePrice')
data.info()
data = data[data.SalePrice.notnull()]
cols = cols.drop(['Alley', 'PoolQC', 'MiscFeature'])
data.head()
categorical = data.select_dtypes(exclude=[np.number])
numerical = data.select_dtypes(include=[np.number])

for (idx, feat) in enumerate(numerical.columns.difference(['Price'])):
    ax = sns.jointplot(x=feat, y='SalePrice', data=numerical, kind='scatter')
    ax.set_axis_labels(feat, 'SalePrice')
data = data.drop(data[data['1stFlrSF'] > 4000].index)
data['SalePrice'].describe()

g = sns.distplot(data['SalePrice'])
for item in g.get_xticklabels():
    item.set_rotation(90)
print('Skewness: %f' % data['SalePrice'].skew())
print('Kurtosis: %f' % data['SalePrice'].kurt())
for (col, col_data) in data.items():
    if is_string_dtype(col_data):
        data[col] = data[col].astype('category').cat.as_ordered().cat.codes
        test[col] = test[col].astype('category').cat.as_ordered().cat.codes
y = data.SalePrice
X = data[cols]
test = test[cols]
params_var = {'max_depth': [1, 2, 3, 4, 5], 'gamma': [0, 0.5, 1], 'n_estimators': randint(1, 1001), 'learning_rate': uniform(), 'subsample': uniform(), 'colsample_bytree': uniform()}
params_fixed = {'silent': 1}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
model = RandomizedSearchCV(estimator=XGBRegressor(**params_fixed, seed=seed), param_distributions=params_var, n_iter=10, cv=cv, scoring='r2', random_state=seed)