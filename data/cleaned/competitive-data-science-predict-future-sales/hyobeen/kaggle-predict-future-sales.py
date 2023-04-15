import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from math import ceil

plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', error_bad_lines=False)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', error_bad_lines=False)
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_c = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
data.head()
data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
data.head()
print(data.dtypes)
data['date_block_num'] = data['date_block_num'].astype(str)
data['shop_id'] = data['shop_id'].astype(str)
data['item_id'] = data['item_id'].astype(str)
print(data.dtypes)
data.describe()
data.apply(lambda x: sum(x.isnull()), axis=0)
data.boxplot(column='item_price')

data['shop_id'].unique()
data['item_id'].unique()
data['date_block_num'].unique()
data['shop_id'].value_counts().plot(kind='bar', figsize=(15, 5))
data['date_block_num'].value_counts().plot(kind='bar', figsize=(15, 5))
data['item_id'].value_counts()
modified = data.pivot_table(index=['shop_id', 'item_id'], values='item_cnt_day', columns='date_block_num', aggfunc='sum').fillna(0.0)
train_df = modified.reset_index()
train_df['shop_id'] = train_df.shop_id.astype('str')
train_df['item_id'] = train_df.item_id.astype('str')
train_df.head()
train_df = train_df[['shop_id', 'item_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33']]
train_df.head()
X_train = train_df.iloc[:, train_df.columns != '33'].values
y_train = train_df.iloc[:, train_df.columns == '33'].values
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=10)