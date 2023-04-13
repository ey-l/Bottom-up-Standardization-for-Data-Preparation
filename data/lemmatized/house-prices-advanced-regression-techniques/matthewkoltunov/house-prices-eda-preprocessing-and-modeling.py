import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_test_data = [_input1, _input0]
print('Training data shape: ', _input1.shape)
print('Test data shape: ', _input0.shape)
_input1.head()
_input1.info()
_input0.head()
_input0.info()
_input1.describe()
_input1.describe(include=['O'])
id_test = _input0['Id'].tolist()
for data in train_test_data:
    data = data.drop(['Id'], axis=1, inplace=False)
print(_input1.shape, _input0.shape)
train_data_num = _input1.select_dtypes(exclude=['object'])
test_data_num = _input0.select_dtypes(exclude=['object'])
train_data_num.head()
train_data_num.hist(figsize=(25, 30), bins=30)
selector = VarianceThreshold(threshold=0.05)