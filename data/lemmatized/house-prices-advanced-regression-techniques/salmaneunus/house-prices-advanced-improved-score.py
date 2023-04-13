import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
all_data = pd.concat((_input1.loc[:, 'MSSubClass':'SaleCondition'], _input0.loc[:, 'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({'price': _input1['SalePrice'], 'log(price + 1)': np.log1p(_input1['SalePrice'])})
prices.hist()
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = _input1[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
all_data
all_data.describe()
sns.lineplot(data=all_data['LotArea'], label='Lot Area')
sns.barplot(x=all_data.index, y=all_data['LotFrontage'])
X_train = all_data[:_input1.shape[0]]
X_test = all_data[_input1.shape[0]:]
y = _input1.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=10))
    return rmse
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title='Change in Root Mean Squared Error')
plt.xlabel('alpha')
plt.ylabel('rmse')
cv_ridge.min()