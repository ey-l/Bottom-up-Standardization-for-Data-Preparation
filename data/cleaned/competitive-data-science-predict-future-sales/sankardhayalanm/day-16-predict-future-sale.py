import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def check_RMSE(y_train, train_prediction, y_test, test_predicition):
    print('Root Mean squared error for the train data  =  ', mean_squared_error(y_train, train_prediction, squared=False))
    print('Root Mean squared error for the test data  =  ', mean_squared_error(y_test, test_predicition, squared=False))
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test.head(2)
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_items.head(2)
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train.head(2)
df_test['date_block_num'] = 34
df_test = df_test[['date_block_num', 'shop_id', 'item_id']]
df_test.head(2)
item_price = dict(df_train.groupby('item_id')['item_price'].last().reset_index().values)
df_test['item_price'] = df_test.item_id.map(item_price)
df_test.head(2)
df_train = df_train[df_train.item_id.isin(df_test.item_id)]
df_train = df_train[df_train.shop_id.isin(df_test.shop_id)]
df_train = df_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'last', 'item_cnt_day': 'sum'}).reset_index()
df_train.head(2)
df_train['shop*item'] = df_train.shop_id * df_train.item_id
df_train.head(2)
df_test['shop*item'] = df_test.shop_id * df_test.item_id
df_test.head(2)
df_items.drop('item_name', axis=1, inplace=True)
item_cat = dict(df_items.values)
df_train['item_cat'] = df_train.item_id.map(item_cat)
df_train.head(2)
df_test['item_cat'] = df_test.item_id.map(item_cat)
df_test.head(2)
df = pd.concat([df_train, df_test])
df.item_price = np.log1p(df.item_price)
df.item_price = df.item_price.fillna(df.item_price.mean())
df.item_cnt_day = df.item_cnt_day.apply(lambda x: 10 if x > 10 else x)
df_train = df[df.item_cnt_day.notnull()]
df_train.head(2)
df_test = df[df.item_cnt_day.isnull()]
df_test.drop('item_cnt_day', axis=1, inplace=True)
df_test.head(2)
X = df_train.drop('item_cnt_day', axis=1).values
y = df_train.item_cnt_day.values
SC = MinMaxScaler()