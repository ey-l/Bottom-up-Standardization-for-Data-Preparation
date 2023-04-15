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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_test_data = [train_data, test_data]
print('Training data shape: ', train_data.shape)
print('Test data shape: ', test_data.shape)
train_data.head()
train_data.info()
test_data.head()
test_data.info()
train_data.describe()
train_data.describe(include=['O'])
id_test = test_data['Id'].tolist()
for data in train_test_data:
    data.drop(['Id'], axis=1, inplace=True)
print(train_data.shape, test_data.shape)
train_data_num = train_data.select_dtypes(exclude=['object'])
test_data_num = test_data.select_dtypes(exclude=['object'])
train_data_num.head()
train_data_num.hist(figsize=(25, 30), bins=30)
selector = VarianceThreshold(threshold=0.05)