import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras as tfk
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
juntos = pd.concat([_input1, _input0])
print(_input0.shape)
print(_input1.shape)
print(juntos.shape)
juntos.head()
_input1.head()
_input1.info()
sns.distplot(_input1['SalePrice'])
total = juntos.isnull().sum().sort_values(ascending=False)
percent = (juntos.isnull().sum() / juntos.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(23)
juntos = juntos.drop(missing_data[missing_data['Total'] > 3].index, 1)
cat_cols = ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
juntos_num = pd.get_dummies(juntos, columns=cat_cols)
treino_num = juntos_num.iloc[0:1460, :]
teste_num = juntos_num.iloc[1460:2920, :]
treino_num = pd.concat([treino_num, _input1['SalePrice']], axis=1)
clf = IsolationForest(max_samples=100, random_state=42)