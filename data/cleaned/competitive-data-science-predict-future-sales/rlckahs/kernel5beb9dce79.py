import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
itemcat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales.loc[sales.item_cnt_day < 0.0, 'item_cnt_day'] = 0
sales_sum_shopitem = sales.groupby(['shop_id', 'item_id', 'item_price', 'date_block_num'], as_index=False).sum()
X = sales_sum_shopitem.iloc[:, 1:-1]
y = sales_sum_shopitem['item_cnt_day']
X_train = sales_sum_shopitem.drop(['item_cnt_day', 'item_price', 'date_block_num'], axis=1)
y_train = sales_sum_shopitem.item_cnt_day
X_test = test.drop('ID', axis=1)
lin_reg = LinearRegression()