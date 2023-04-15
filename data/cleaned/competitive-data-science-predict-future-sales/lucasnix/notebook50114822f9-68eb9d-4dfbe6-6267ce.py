import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categoria = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items_full = items.merge(item_categoria, on='item_category_id', how='left')
len(sales[sales.item_cnt_day < 0])
submissao = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
teste = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_full = sales.merge(shops, on='shop_id', how='left').merge(items_full, on='item_id', how='left')
sales_full['date'] = pd.to_datetime(sales_full['date'], format='%d.%m.%Y')
sales_full['month'] = sales_full['date'].dt.to_period('M')
sales_full = sales_full.groupby(['month', 'shop_id', 'shop_name', 'item_category_id', 'item_category_name', 'item_id', 'item_name']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
sales_full['item_category_name'] = sales_full['item_category_name'].str.strip()
sales_full['item_major_category'] = sales_full['item_category_name'].apply(lambda x: x.split('-')[0].strip())
sales_full['item_sales_day'] = sales_full['item_price'] * sales_full['item_cnt_day']
sales_full.head()
sales_full.month.max()
base_quero = sales_full[sales_full.month == '2015-10']
shops = base_quero.shop_id.unique()
sales_full = sales_full[sales_full.shop_id.isin(list(shops))]
len(sales_full[~sales_full.shop_id.isin(list(shops))])
sales_full.groupby('month').sum('item_cnt_day').sort_values('month').plot.line(y='item_cnt_day')
sales_full['item_name'].value_counts()
sales_full.drop('month', axis=1).groupby('shop_name').apply(lambda df: df.sort_values(by='item_cnt_day', ascending=False))
sales_full.groupby(['shop_name', 'item_major_category']).apply(lambda df: df.sort_values(by='item_cnt_day', ascending=False))
sales_clean = sales_full[['item_id', 'shop_name', 'item_cnt_day', 'item_major_category']]
pd.set_option('display.max_rows', 100)
sales_clean.groupby(['shop_name', 'item_major_category']).sum('item_cnt_day').sort_values(by='item_cnt_day', ascending=False)
sales_full.head()
submissao.head()
teste.head()
from catboost import CatBoostRegressor
X = sales_full.rename({'item_cnt_day': 'item_cnt_month', 'item_sales_day': 'item_sales_month'}, axis=1)
X = X[X.columns[~X.columns.str.contains('name')]]
X = X.drop('item_major_category', axis=1)
X['year_month'] = X['month']
X['month'] = X['year_month'].dt.month
X['year'] = X['year_month'].dt.year
y = X[['year_month', 'item_cnt_month']]
X.drop(['item_cnt_month', 'item_sales_month'], axis=1, inplace=True)
cut_date = '2015-08'
X_train = X[X.year_month < cut_date]
X_test = X[X.year_month >= cut_date]
y_train = y[y.year_month < cut_date]
y_test = y[y.year_month >= cut_date]
X_train.head()
y_train.head()
X_train = X_train.drop('year_month', axis=1)
X_test = X_test.drop('year_month', axis=1)
y_train = y_train.drop('year_month', axis=1)
model = CatBoostRegressor()