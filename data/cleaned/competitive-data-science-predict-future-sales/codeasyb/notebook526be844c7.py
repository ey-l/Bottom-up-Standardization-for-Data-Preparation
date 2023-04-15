import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sales = '_data/input/competitive-data-science-predict-future-sales/sales_train.csv'
test = '_data/input/competitive-data-science-predict-future-sales/test.csv'
sample = '_data/input/competitive-data-science-predict-future-sales/sample_submission.csv'
df_train = pd.read_csv(sales)
print(df_train.shape)
df_train.head()
df_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.strftime('%Y-%m'))
df_train.head(3)
df = df_train.groupby(['date', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df_test = pd.read_csv(test)
df_test = pd.merge(df_test, df, on=['shop_id', 'item_id'], how='left')
df_test.drop(['ID', '2013-01'], axis=1, inplace=True)
df_test = df_test.fillna(0)
df_test.head(5)
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis=1)
X_test = df_test
print('\n-------------------------------')
print('Our Dataframes dimensionalities')
print('-------------------------------')
print('Data DataFrame: {0}\nTarget Values:  {1}\nTest Dataframe: {2}'.format(X_train.shape, Y_train.shape, X_test.shape))
print('-------------------------------\n')
print('----------------------------------')
print('Starting the training phase')
print('----------------------------------')
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set:  ', x_test.shape, y_test.shape)
print('----------------------------------\n')
print('|vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv|')
print('|           --------------------------              |')
print('|           Linear Regression accuracy              |')
print('|           --------------------------              |')
from sklearn.linear_model import LinearRegression
LR = LinearRegression()