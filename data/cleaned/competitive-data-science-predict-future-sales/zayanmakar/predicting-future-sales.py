import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
Id = test['ID']
test['date_block_num'] = 0
test = test[['date_block_num', 'shop_id', 'item_id']]
item_price = dict(train.groupby('item_id')['item_price'].last().reset_index().values)
test['item_price'] = test.item_id.map(item_price)
test['item_price'] = test['item_price'].fillna(test['item_price'].median())
X = train.drop(['item_cnt_day'], axis=1)
y = train['item_cnt_day']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=44, shuffle=True)
lrm = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=33)