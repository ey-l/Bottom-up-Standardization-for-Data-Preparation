import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0.head()
_input2.head()
_input0['date'] = _input0['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
print(_input0.dtypes)
_input0['date_block_num'] = _input0['date_block_num'].astype(str)
_input0['shop_id'] = _input0['shop_id'].astype(str)
_input0['item_id'] = _input0['item_id'].astype(str)
print(_input0.dtypes)
_input0.describe()
_input0.apply(lambda x: sum(x.isnull()), axis=0)
_input0.boxplot(column='item_price')
_input0['shop_id'].unique()
_input0['item_id'].unique()
_input0['date_block_num'].unique()
_input0['shop_id'].value_counts().plot(kind='bar', figsize=(15, 5))
_input0['date_block_num'].value_counts().plot(kind='bar', figsize=(15, 5))
_input0['item_id'].value_counts()
modified = _input0.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', aggfunc='sum').fillna(0.0)
train_df = modified.reset_index()
train_df['shop_id'] = train_df.shop_id.astype('str')
train_df['item_id'] = train_df.item_id.astype('str')
train_df.head()
train_df = train_df[['shop_id', 'item_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33']]
train_df.head()
X_train = train_df.iloc[:, train_df.columns != '33'].values
y_train = train_df.iloc[:, train_df.columns == '33'].values
rf = RandomForestRegressor(random_state=10)