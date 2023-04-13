import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('shape of train data', _input1.shape)
print('shape of test data', _input0.shape)
_input1.columns
print('First five rows of Train Dataset')
_input1.head()
_input1.info()
_input1.isnull().sum().sort_values(ascending=False)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
Id_test_list = _input0['Id'].tolist()
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
print("new shape of train data after dropping 'Id' column", _input1.shape)
print("new shape of test data after dropping 'Id' column", _input0.shape)
train_df_num = _input1.select_dtypes(np.number)
train_df_num.shape
profile_num = ProfileReport(train_df_num, 'Train Numerical Datatype Profiling Report')
profile_num.to_widgets()
train_df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.plot()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
train_df_dirty = _input1.copy()
test_df_dirty = _input0.copy()
categorical_column = train_df_dirty.describe(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder()
train_df_dirty[categorical_column] = ordinal_encoder.fit_transform(train_df_dirty[categorical_column])
categorical_column = test_df_dirty.describe(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder()
test_df_dirty[categorical_column] = ordinal_encoder.fit_transform(test_df_dirty[categorical_column])
train_df_dirty.isnull().sum().sort_values(ascending=False)[:20]
my_imputer = SimpleImputer(strategy='median')
train_df_dirty_imputed = pd.DataFrame(my_imputer.fit_transform(train_df_dirty))
train_df_dirty_imputed.columns = train_df_dirty.columns
train_df_dirty_imputed.isnull().sum().sort_values(ascending=False)
my_imputer = SimpleImputer(strategy='median')
test_df_dirty_imputed = pd.DataFrame(my_imputer.fit_transform(test_df_dirty))
test_df_dirty_imputed.columns = test_df_dirty.columns
X = train_df_dirty_imputed.drop(['SalePrice'], axis=1)
y = train_df_dirty_imputed['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=100)
lr = LinearRegression()