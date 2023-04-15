import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.shape
df_test.shape
df_train.info()
df_train.isnull().sum()
lst = df_train.columns
nan_cols = []
for i in lst:
    if df_train[i].isnull().sum() > 0:
        nan_cols.append(i)
print(nan_cols)
lst1 = df_test.columns
nan_cols1 = []
for i in lst1:
    if df_test[i].isnull().sum() > 0:
        nan_cols1.append(i)
print(nan_cols1)
nan_train_num_cols = [col for col in nan_cols if df_train[col].dtypes != 'O']
nan_train_cat_cols = [col for col in nan_cols if df_train[col].dtypes == 'O']
nan_test_num_cols1 = [col for col in nan_cols1 if df_test[col].dtypes != 'O']
nan_test_cat_cols1 = [col for col in nan_cols1 if df_test[col].dtypes == 'O']
for j in nan_train_num_cols:
    df_train[j].fillna(df_train[j].mean(), inplace=True)
for k in nan_train_cat_cols:
    df_train[k].fillna(df_train[k].mode()[0], inplace=True)
for j in nan_test_num_cols1:
    df_test[j].fillna(df_test[j].mean(), inplace=True)
for k in nan_test_cat_cols1:
    df_test[k].fillna(df_test[k].mode()[0], inplace=True)
df_train.head()
cat_cols = [cols for cols in df_train.columns if df_train[cols].dtypes == 'O']
cat_cols1 = [cols for cols in df_test.columns if df_test[cols].dtypes == 'O']
cat_cols == cat_cols1
X = df_train.drop('SalePrice', axis=1)
Y = df_train['SalePrice']
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
ct = ColumnTransformer([('step1', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)])
pipeline = Pipeline([('cltf_step', ct), ('Gradient Boost', XGBRegressor(learning_rate=1, random_state=42, n_jobs=5))])