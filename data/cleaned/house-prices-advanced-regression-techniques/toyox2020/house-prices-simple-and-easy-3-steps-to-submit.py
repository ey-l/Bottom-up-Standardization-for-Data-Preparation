import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test.head()
print('=========== train infomation ===========')
train.info()
print('\n\n=========== test infomation ===========')
test.info()
data = pd.concat([train, test])
data.shape
numerics = data.loc[:, data.dtypes != 'object'].drop('Id', axis=1)
numerics.head()
log_numerics = np.log1p(numerics)
log_numerics.head()
skewness = pd.concat([numerics.apply(lambda x: skew(x.dropna())), log_numerics.apply(lambda x: skew(x.dropna()))], axis=1).rename(columns={0: 'original', 1: 'logarithmization'}).sort_values('original')
skewness.plot.barh(figsize=(12, 10), title='Comparison of skewness of original and logarithmized', width=0.8)
cat_cols = data.loc[:, data.dtypes == 'object'].columns
data.loc[:, cat_cols].head()
cat_data = pd.get_dummies(data.loc[:, cat_cols], drop_first=True)
cat_data.head()
optimized_data = pd.concat([data['Id'], cat_data, log_numerics], axis=1)
optimized_data.head()
train = optimized_data[:train.shape[0]]
test = optimized_data[train.shape[0]:].drop(['Id', 'SalePrice'], axis=1)
X_train = train.drop(['Id', 'SalePrice'], axis=1)
y_train = train['SalePrice']
lgb_train = lgb.Dataset(X_train, y_train)
params = {'objective': 'regression', 'metric': {'rmse'}}
gbm = lgb.train(params, lgb_train)
pred = gbm.predict(test)
pred = np.expm1(pred)
results = pd.Series(pred, name='SalePrice')
submission = pd.concat([submission['Id'], results], axis=1)

submission.head()