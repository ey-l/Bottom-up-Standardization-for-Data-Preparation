import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import scipy.stats
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_sample_sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train.head()
df_train.columns
df_sample_sub.head()
df_train.info()
df_test.info()
nan_columns_train = df_train.isna().sum().sort_values(ascending=False)
nan_columns_test = df_test.isna().sum().sort_values(ascending=False)
too_many_nans = nan_columns_test[:4].index.tolist()
too_many_nans
df_train.drop(columns=too_many_nans, inplace=True)
df_test.drop(columns=too_many_nans, inplace=True)
df_train.corr()
correlation_price = np.abs(df_train.corr()['SalePrice']).sort_values()
correlation_price
columns_to_drop = list(correlation_price[correlation_price < 0.022].index)
columns_to_drop
df_train.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)
correlation = df_train.corr().abs()
unstack_corr = correlation.unstack()
correlated_pairs = unstack_corr.sort_values(kind='quicksort', ascending=False)
high_corr_pairs = correlated_pairs[(correlated_pairs < 1.0) & (correlated_pairs >= 0.8)]
high_corr_pairs
for col_name in high_corr_pairs.index.get_level_values(0):
    print(col_name)
    print(df_train[col_name].isna().sum())
    print(df_test[col_name].isna().sum())
null_garage_train = df_train.GarageYrBlt.isna()
null_garage_test = df_train.GarageYrBlt.isna()
null_garate_ind_train = null_garage_train[null_garage_train == True].index
null_garate_ind_test = null_garage_test[null_garage_test == True].index
df_train.GarageYrBlt.iloc[null_garate_ind_train] = df_train.YearBuilt.iloc[null_garate_ind_train]
df_test.GarageYrBlt.iloc[null_garate_ind_test] = df_test.YearBuilt.iloc[null_garate_ind_test]
ax = plt.figure(figsize=(20, 20))
df_train.hist(ax=ax)
num_train = df_train.select_dtypes(include=['number']).dropna(axis=0)
train_data = num_train.iloc[:, :-1]
target = num_train.iloc[:, -1]
reg = LinearRegression()
scaler = StandardScaler()
pipeline = Pipeline([('scaler', scaler), ('reg', reg)])