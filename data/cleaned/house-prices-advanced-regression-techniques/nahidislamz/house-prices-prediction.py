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
from sklearn.impute import SimpleImputer
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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
sum(train.isna().sum())
sum(test.isna().sum())

def fill_value(data):
    for name in data.columns:
        x = data[name].isna().sum()
        if x > 0:
            val_list = np.random.choice(data.groupby(name).count().index, x, p=data.groupby(name).count()['Id'].values / sum(data.groupby(name).count()['Id'].values))
            train.loc[data[name].isna(), name] = val_list
fill_value(train)
for name in test.columns:
    x = test[name].isna().sum()
    if x > 0:
        val_list = np.random.choice(test.groupby(name).count().index, x, p=test.groupby(name).count()['Id'].values / sum(test.groupby(name).count()['Id'].values))
        test.loc[test[name].isna(), name] = val_list
sum(train.isna().sum())
sum(test.isna().sum())
train_df = train.drop('SalePrice', axis=1)
data = pd.concat([train_df, test])
le = preprocessing.LabelEncoder()
for name in data.columns:
    if data[name].dtypes == 'O':
        print(name)
        data[name] = data[name].astype(str)
        train[name] = train[name].astype(str)
        test[name] = test[name].astype(str)