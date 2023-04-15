import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_items.head(5)
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test.head(5)
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train.head(5)
df_item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_item_categories.head(5)
df = df_train
df = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
df.head(5)
df_test['date_block_num'] = 34
item_price = dict(df.groupby('item_id')['item_price'].last().reset_index().values)
df_test['item_price'] = df_test['item_id'].map(item_price)
df_test['item_price'] = df_test['item_price'].fillna(df_test['item_price'].median())
df_test.head(5)
df = df.sample(frac=1)
np.array(df.drop(['item_cnt_day'], 1))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x = np.array(df.drop(['item_cnt_day'], 1))
y = np.array(df.iloc[:, 4])
SC = MinMaxScaler()