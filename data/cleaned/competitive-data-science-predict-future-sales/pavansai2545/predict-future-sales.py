import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
d_item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
d_item_cat.head()
d_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
d_items.head()
d_sam_sub = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
d_sam_sub.head()
d_sal_tra = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
d_sal_tra.head()
d_shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
d_shop.head()
d_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
d_test.head()
dt_test = d_test.drop(labels='ID', axis=1)
dt_test
print('items categorical null values', d_item_cat.isnull().sum().sum())
print('items  null values', d_items.isnull().sum().sum())
print('sales train null values', d_sal_tra.isnull().sum().sum())
print('test data null values', d_test.isnull().sum().sum())
print('items categorical unique values', d_item_cat.nunique())
print('items  unique values', d_items.nunique())
print('sales train unique values', d_sal_tra.nunique())
print('test data unique values', d_test.nunique())
new_data = d_sal_tra.copy()
new_data
new_data = pd.merge(d_item_cat, d_items, how='inner', on='item_category_id')
new_data
new_data = pd.merge(new_data, d_sal_tra, how='inner', on='item_id')
new_data
new_data = pd.merge(new_data, d_shop, how='inner', on='shop_id')
new_data
dx = ['shop_id', 'item_id']
x = new_data[dx]
y = new_data['item_cnt_day']
(x, y)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=20)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()