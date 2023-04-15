import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
path_data = '_data/input/house-prices-advanced-regression-techniques/'
df_train = pd.read_csv(path_data + 'train.csv')
df_test = pd.read_csv(path_data + 'test.csv')
df_train.head(5)
df_train['SalePrice'].describe()
plt.figure(figsize=(12, 10))
plt.hist(df_train['SalePrice'], bins=25)
df_nulls = df_train.isnull().sum().sort_values(ascending=False)
df_nulls
df_ratio_nulls = (df_nulls / len(df_train)).reset_index()
df_ratio_nulls.columns = ['Feature', 'ratio_nulls']
df_ratio_nulls[df_ratio_nulls['ratio_nulls'] > 0]
cat_cols = ['MSSubClass', 'YrSold', 'MoSold']
for col in cat_cols:
    df_train[col] = df_train[col].astype(str)
df_train[cat_cols]
num_cols = ['GarageArea', 'GarageCars', 'MasVnrArea']
for col in num_cols:
    df_train[col] = df_train[col].fillna(0)
list_selected_vars = ['PoolArea', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtUnfSF', 'TotalBsmtSF', 'OverallQual', 'GrLivArea']
list_target_col = ['SalePrice']
df_train_complete = df_train.copy()
df_train = df_train[list_selected_vars + list_target_col]
for var in list_selected_vars:
    if len(df_train[var].unique()) <= 20:
        (f, ax) = plt.subplots(figsize=(12, 8))
        plt.title(f'Variable: {var}', fontsize=14)
        sns.boxplot(x=var, y='SalePrice', data=df_train)
    else:
        (f, ax) = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x=var, y='SalePrice', data=df_train)
target_col = 'SalePrice'
corr_mat = df_train_complete.corr()
(f, ax) = plt.subplots(figsize=(16, 10))
sns.heatmap(corr_mat)
(df_train, df_val) = train_test_split(df_train, test_size=0.15, random_state=12)

def get_metric_competition_error(y_true, y_pred):
    y_true = np.log1p(y_true)
    y_pred = np.log1p(y_pred)
    msle = mean_squared_error(y_true, y_pred)
    rmsle = np.sqrt(msle)
    return rmsle
sale_price_mean = df_train['SalePrice'].mean()
train_rmsle = get_metric_competition_error(df_train['SalePrice'], [sale_price_mean for _ in range(len(df_train))])
val_rmsle = get_metric_competition_error(df_val['SalePrice'], [sale_price_mean for _ in range(len(df_val))])
print(f'Train Metric: {train_rmsle}, Validation Metric: {val_rmsle}')
model_linear = LinearRegression()