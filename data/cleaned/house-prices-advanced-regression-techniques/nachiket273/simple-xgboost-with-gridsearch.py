import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
(train_df.shape, test_df.shape)
train_df['SalePrice'].describe()
sns.distplot(train_df['SalePrice'])
labels = train_df['SalePrice']
labels
train_df.drop(['SalePrice'], axis=1, inplace=True)
train_idx = train_df.shape[0]
train_idx
new_df = pd.concat([train_df, test_df], ignore_index=True, axis=0)
new_df.head()
new_df.shape
nan_pct = [new_df[col].isna().sum() * 100 / new_df.shape[0] for col in new_df.columns]
nan_df = pd.DataFrame(columns=['column', 'percentage'])
nan_df['column'] = new_df.columns
nan_df['percentage'] = nan_pct
nan_df[nan_df['percentage'] > 50]
(new_df['Alley'].unique(), new_df['PoolQC'].unique(), new_df['Fence'].unique(), new_df['MiscFeature'].unique())
new_df.drop(nan_df[nan_df['percentage'] > 50]['column'].unique(), axis=1, inplace=True)
nan_df[(nan_df['percentage'] < 50) & (nan_df['percentage'] > 5)]
new_df['FireplaceQu'].unique()
new_df.drop(['FireplaceQu'], axis=1, inplace=True)
nan_feel_cols = nan_df[(nan_df['percentage'] < 45) & (nan_df['percentage'] > 0)]['column'].unique()
obj_cols = new_df[nan_feel_cols].select_dtypes(include=['object']).columns
float_cols = new_df[nan_feel_cols].select_dtypes(include=['float']).columns
new_df[obj_cols] = new_df[obj_cols].fillna('None')
for col in float_cols:
    new_df[col] = new_df[col].fillna(new_df[col].mode().iloc[0])
new_df.head()
sns.countplot(new_df['YrSold'])
sns.distplot(new_df['YearBuilt'])
sns.distplot(new_df['LotFrontage'])
sns.distplot(new_df['LotArea'])
for obj in obj_cols:
    print(obj)
    print(new_df[obj].unique())
new_df_1 = pd.get_dummies(new_df)
new_df_1.head()
train_df_1 = new_df_1[:train_idx]
test_df_1 = new_df_1[train_idx:]
rf = RandomForestRegressor(max_depth=40, min_samples_leaf=3, min_samples_split=8, n_estimators=5000, random_state=17)