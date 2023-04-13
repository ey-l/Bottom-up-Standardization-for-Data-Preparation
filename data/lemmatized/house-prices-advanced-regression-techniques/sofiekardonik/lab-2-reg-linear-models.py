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
all_data
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({'price': _input1['SalePrice'], 'log(price + 1)': np.log1p(_input1['SalePrice'])})
prices.hist()
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = _input1[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
numeric_feats
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:_input1.shape[0]]
X_test = all_data[_input1.shape[0]:]
y = _input1.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))
    return rmse