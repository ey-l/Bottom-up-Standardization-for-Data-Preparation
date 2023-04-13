import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import numpy as np
import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
sum(_input1.isna().sum())
sum(_input0.isna().sum())
for name in _input1.columns:
    x = _input1[name].isna().sum()
    if x > 0:
        val_list = np.random.choice(_input1.groupby(name).count().index, x, p=_input1.groupby(name).count()['Id'].values / sum(_input1.groupby(name).count()['Id'].values))
        _input1.loc[_input1[name].isna(), name] = val_list
for name in _input0.columns:
    x = _input0[name].isna().sum()
    if x > 0:
        val_list = np.random.choice(_input0.groupby(name).count().index, x, p=_input0.groupby(name).count()['Id'].values / sum(_input0.groupby(name).count()['Id'].values))
        _input0.loc[_input0[name].isna(), name] = val_list
sum(_input1.isna().sum())
sum(_input0.isna().sum())
train_df = _input1.drop('SalePrice', axis=1)
data = pd.concat([train_df, _input0])
le = preprocessing.LabelEncoder()
for name in data.columns:
    if data[name].dtypes == 'O':
        print(name)
        data[name] = data[name].astype(str)
        _input1[name] = _input1[name].astype(str)
        _input0[name] = _input0[name].astype(str)