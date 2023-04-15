import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
df
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train = train.merge(items, on='item_id')
train['item_category_id'] = train['item_category_id'].astype('category')
train.columns
test.columns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
X = train[['shop_id', 'item_id']]
y = train[['item_cnt_day']]