import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales_train.info()
type(sales_train['date'])
sales_train.head()
test.tail()
print(sorted(sales_train['shop_id'].unique()))
print(sorted(test['shop_id'].unique()))
gp = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
X = np.array(list(map(list, gp.index.values)))
y_train = gp.values
test['date_block_num'] = sales_train['date_block_num'].max() + 1
X_test = test[['date_block_num', 'shop_id', 'item_id']].values