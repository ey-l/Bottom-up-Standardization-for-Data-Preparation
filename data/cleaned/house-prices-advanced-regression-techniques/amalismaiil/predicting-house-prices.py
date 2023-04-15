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
housing = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
housing_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
categorical = []
for i in housing.columns:
    if housing[i].dtype == object:
        categorical.append(i)
categorical
housing = housing.drop(columns=categorical)
housing.info()
total = housing.isnull().sum().sort_values(ascending=False)
percent = (housing.isnull().sum() / housing.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
housing = housing.fillna(housing.median())
housing.isnull().sum()
corr = housing.corr()
corr
plt.figure(figsize=(25, 25))
sns.heatmap(corr, annot=True)

housing = housing.drop(columns=['GarageYrBlt', '1stFlrSF', 'GarageArea']).set_index('Id')
housing
corr_price = housing.corr()['SalePrice'].sort_values(ascending=False)
corr_price
skewed_col = []
for i in housing.columns:
    if housing[i].skew() > 0.5:
        skewed_col.append(i)
len(skewed_col)
housing[skewed_col] = housing[skewed_col].apply(lambda i: np.log1p(i))
x = housing.iloc[:, :-1]
y = housing.iloc[:, -1]
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
housing_test = housing_test.drop(columns=categorical)
housing_test = housing_test.drop(columns=['GarageYrBlt', '1stFlrSF', 'GarageArea']).set_index('Id')
total_test = housing_test.isnull().sum().sort_values(ascending=False)
percent_test = (housing_test.isnull().sum() / housing.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(10)
housing_test = housing_test.fillna(housing_test.median())
skewed_col_test = []
for i in housing_test.columns:
    if housing_test[i].skew() > 0.5:
        skewed_col_test.append(i)
skewed_col_test
housing_test[skewed_col[:-1]] = housing_test[skewed_col[:-1]].apply(lambda i: np.log1p(i))
housing_test = sc.transform(housing_test)
final = SVR(kernel='linear', C=1)