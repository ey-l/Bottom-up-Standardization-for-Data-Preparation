import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
categorical = []
for i in _input1.columns:
    if _input1[i].dtype == object:
        categorical.append(i)
categorical
_input1 = _input1.drop(columns=categorical)
_input1.info()
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
_input1 = _input1.fillna(_input1.median())
_input1.isnull().sum()
corr = _input1.corr()
corr
plt.figure(figsize=(25, 25))
sns.heatmap(corr, annot=True)
_input1 = _input1.drop(columns=['GarageYrBlt', '1stFlrSF', 'GarageArea']).set_index('Id')
_input1
corr_price = _input1.corr()['SalePrice'].sort_values(ascending=False)
corr_price
skewed_col = []
for i in _input1.columns:
    if _input1[i].skew() > 0.5:
        skewed_col.append(i)
len(skewed_col)
_input1[skewed_col] = _input1[skewed_col].apply(lambda i: np.log1p(i))
x = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
sns.distplot(y)
sc = StandardScaler()
x = sc.fit_transform(x)
lr = LinearRegression()
cv_lr = cross_validate(lr, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_lr['test_score'].mean()
svr = SVR(kernel='linear', C=1)
cv_svr = cross_validate(svr, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_svr['test_score'].mean()
sgd = SGDRegressor()
cv_sgd = cross_validate(sgd, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_sgd['test_score'].mean()
ridge = Ridge()
cv_r = cross_validate(ridge, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_r['test_score'].mean()
lasso = Lasso(alpha=0.001)
cv_l = cross_validate(lasso, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_l['test_score'].mean()
en = ElasticNet(alpha=0.001)
cv_en = cross_validate(en, x, y, cv=10, scoring='neg_root_mean_squared_error')
cv_en['test_score'].mean()
_input0 = _input0.drop(columns=categorical)
_input0 = _input0.drop(columns=['GarageYrBlt', '1stFlrSF', 'GarageArea']).set_index('Id')
total_test = _input0.isnull().sum().sort_values(ascending=False)
percent_test = (_input0.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(10)
_input0 = _input0.fillna(_input0.median())
skewed_col_test = []
for i in _input0.columns:
    if _input0[i].skew() > 0.5:
        skewed_col_test.append(i)
skewed_col_test
_input0[skewed_col[:-1]] = _input0[skewed_col[:-1]].apply(lambda i: np.log1p(i))
_input0 = sc.transform(_input0)
final = SVR(kernel='linear', C=1)