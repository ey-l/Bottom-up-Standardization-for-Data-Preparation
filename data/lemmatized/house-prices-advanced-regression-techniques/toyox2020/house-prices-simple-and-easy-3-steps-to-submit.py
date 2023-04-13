import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input2.head()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
print('=========== train infomation ===========')
_input1.info()
print('\n\n=========== test infomation ===========')
_input0.info()
data = pd.concat([_input1, _input0])
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
_input1 = optimized_data[:_input1.shape[0]]
_input0 = optimized_data[_input1.shape[0]:].drop(['Id', 'SalePrice'], axis=1)
X_train = _input1.drop(['Id', 'SalePrice'], axis=1)
y_train = _input1['SalePrice']
lgb_train = lgb.Dataset(X_train, y_train)
params = {'objective': 'regression', 'metric': {'rmse'}}
gbm = lgb.train(params, lgb_train)
pred = gbm.predict(_input0)
pred = np.expm1(pred)
results = pd.Series(pred, name='SalePrice')
_input2 = pd.concat([_input2['Id'], results], axis=1)
_input2.head()