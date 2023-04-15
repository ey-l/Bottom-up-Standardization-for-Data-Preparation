import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
plt.rcParams['font.size'] = 14
random_seed = 123
from datetime import datetime
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
df1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=[0])
from datetime import datetime
f1 = '%d.%m.%Y'
my_parser = lambda date: pd.datetime.strptime(date, f1)
df1m = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', index_col=[0], parse_dates=[0], date_parser=my_parser)
df1m = df1m.reset_index()
df2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df6 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
df1m3 = pd.merge(df1m, df3, on='item_id', how='left')
df1m32 = pd.merge(df1m3, df2, on='item_category_id', how='left')
df1m324 = pd.merge(df1m32, df4, on='shop_id', how='left')
DF1 = df1m324.groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).sort_values(by=['date_block_num', 'shop_id', 'item_id']).reset_index()
DF2 = df1m324[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].sort_values(['date', 'shop_id', 'item_id']).reset_index()
DF3 = DF2.groupby(['date']).sum()['item_cnt_day']
DF4 = DF2.rename(columns={'date': 'ds', 'item_cnt_day': 'y'})
DF4 = DF2.rename(columns={'date': 'ds', 'item_cnt_day': 'y'}).groupby(['ds']).sum()['y'].reset_index()
DF5 = DF4[['ds', 'y']]
DF5 = pd.DataFrame(DF5)
plt.plot(DF5['ds'], DF5['y'])
mday = pd.to_datetime('2015-08-01')
train_index = DF5['ds'] < mday
test_index = DF5['ds'] >= mday
x_train = DF5[train_index]
x_test = DF5[test_index]
dates_test = DF5['ds'][test_index]