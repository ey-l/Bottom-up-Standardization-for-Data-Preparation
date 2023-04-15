import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from xgboost import XGBRegressor

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_sub = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')

def df_head(df):
    return df.head()
df_head(df_sales)
df_head(df_sub)
df_head(df_items)
df_head(df_shops)
df_head(df_test)
df_sales.describe()
df_test.describe().T
df_sales['item_price']
df_sales[['shop_id', 'item_id', 'item_price', 'item_cnt_day']].corr()
df_items.describe().T
plt.figure(figsize=(8, 5))
plt.hist(df_sales['item_id'])

plt.figure(figsize=(8, 5))
plt.hist(df_items['item_category_id'])

plt.figure(figsize=(8, 5))
plt.hist(df_sales['item_cnt_day'])

df_items.groupby('item_category_id').count()
df_items.groupby('item_category_id').mean()
df_items['diff_col_of_item_id'] = df_items.groupby('item_category_id')['item_id'].max() - df_items.groupby('item_category_id')['item_id'].min()
df_items.head()
df_items.head()
df_sales.head()
df_sales.isnull().sum()
df_sales.drop_duplicates(keep='first', inplace=True, ignore_index=True)
df_sales.head()
df_sales[df_sales['item_price'] < 0]
df_sales.drop(df_sales[df_sales['item_cnt_day'] < 0].index, inplace=True)
df_sales.drop(df_sales[df_sales['item_price'] < 0].index, inplace=True)
df_sales.shape
Q1 = np.percentile(df_sales['item_price'], 25.0)
Q3 = np.percentile(df_sales['item_price'], 75.0)
IQR = Q3 - Q1
df_sub1 = df_sales[df_sales['item_price'] > Q3 + 1.5 * IQR]
df_sub2 = df_sales[df_sales['item_price'] < Q1 - 1.5 * IQR]
df_sales.drop(df_sub1.index, inplace=True)
df_sales.shape
df_sales['date_block_num'].unique()
df_sales.groupby('date_block_num')['item_id'].mean()
price = round(np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean(), 2)
print(price)
dict(round(df_sales.groupby('date_block_num')['item_price'].mean(), 4))
df_sales.head()
df_test.head()
replace_dict = dict(round(df_sales.groupby('date_block_num')['item_price'].mean(), 2))
df_sales['date_block_num'] = df_sales['date_block_num'].replace(replace_dict)
df_train = df_sales.copy()
df_train.drop(['date', 'item_price'], axis=1, inplace=True)
df_train.rename(columns={'date_block_num': 'mean_price_by_column'}, inplace=True)
df_train.head()
mean_price = np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean()
mean_price
df_test.shape
df_train.shape
df_test.head()
com_df = pd.concat([df_train, df_test])
com_df['mean_price_by_column'] = com_df['mean_price_by_column'].fillna(value=price)
com_df['item_cnt_day'] = com_df['item_cnt_day'].fillna(value=0)
test_df = com_df[com_df['item_cnt_day'] == 0]
train_df = com_df[com_df['item_cnt_day'] != 0]
test_df.shape
testdf = test_df.copy()
testdf.drop('ID', inplace=True, axis=1)
testdf.drop('item_cnt_day', inplace=True, axis=1)
testdf
traindf = train_df.copy()
traindf.drop('ID', inplace=True, axis=1)
traindf.head()
testdf['item_id'] = (testdf['item_id'] - testdf['item_id'].mean()) / testdf['item_id'].std()
testdf.head()
traindf['item_id'] = (traindf['item_id'] - traindf['item_id'].mean()) / traindf['item_id'].std()
traindf.head()
X = traindf.loc[:, ['mean_price_by_column', 'shop_id', 'item_id']]
y = traindf.loc[:, 'item_cnt_day']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, random_state=42)
model1 = LinearRegression()