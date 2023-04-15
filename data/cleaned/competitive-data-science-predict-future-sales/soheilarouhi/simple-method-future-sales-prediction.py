import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestRegressor
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(sales.info())
print('Number of missing values in dataset : ' + str(sales.isnull().sum().max()))
print(sales.describe())
neg_price = sales.item_price < 0
print(sales[neg_price])
sales.drop(sales[neg_price].index, axis=0, inplace=True)
sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales['year'] = sales.date.dt.year
sales['month'] = sales.date.dt.month
print(sales.head())
print(f'Time range is between {sales.date.min()}, {sales.date.max()}')
sales_total_month = sales.loc[:, ['date_block_num', 'item_cnt_day']].groupby('date_block_num').sum()
sales_total_month.rename(columns={'item_cnt_day': 'total_cnt_month'}, inplace=True)
sales_total_month.plot()
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(x=12, color='red', linestyle='--')
plt.axvline(x=24, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')
legend_x = 1
legend_y = 0.5
plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.ylabel('Total number of sold items per month')
plt.xlabel('Month block number')

price_average_month = sales.loc[:, ['date_block_num', 'item_price']].groupby('date_block_num').mean()
price_average_month.rename(columns={'item_price': 'monthly_average_item_price'}, inplace=True)
price_average_month.plot()
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(x=12, color='red', linestyle='--')
plt.axvline(x=24, color='red', linestyle='--')
plt.axvline(x=36, color='red', linestyle='--')
legend_x = 1
legend_y = 0.5
plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.ylabel('Monthly average price of items')
plt.xlabel('Month block number')

plt.figure(figsize=(20, 14))
sns.countplot(x='item_category_id', data=items)

items = items.drop(columns='item_name')
sales_cat = pd.merge(sales, items, on=['item_id'], how='left')
print(sales_cat.head())
sales_total_cat = sales_cat.loc[:, ['item_category_id', 'item_cnt_day']].groupby('item_category_id').sum()
sales_total_cat.rename(columns={'item_cnt_day': 'total_sold_items'}, inplace=True)
print(sales_total_cat.head())
sales_total_cat.plot()
plt.xlabel('Category ID')
plt.ylabel('Total sold items')

sales_price_per_shop = sales.loc[:, ['shop_id', 'item_price']].groupby('shop_id').mean()
sales_price_per_shop.rename(columns={'item_price': 'average_item_price'}, inplace=True)
sales_price_per_shop.plot(kind='bar', figsize=(20, 14))
plt.xlabel('shop_id')

sales_sold_items_per_shop = sales.loc[:, ['shop_id', 'item_cnt_day']].groupby('shop_id').sum()
sales_sold_items_per_shop.rename(columns={'item_cnt_day': 'total_sold_items'}, inplace=True)
sales_sold_items_per_shop.plot(kind='bar', figsize=(20, 14))

sales_id = pd.merge(sales_cat, test, on=['item_id', 'shop_id'], how='left')
sales_id_useful = sales_id.dropna()
print(sales_id_useful.columns)
sales_monthly = sales_id_useful.groupby(['ID', 'shop_id', 'item_id', 'item_category_id', 'date_block_num', 'year', 'month'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
sales_monthly_sort_id = sales_monthly.sort_values(['ID'], ascending=True).reset_index().drop(columns='index')
sales_monthly_sort_id['ID'] = sales_monthly_sort_id['ID'].astype(int)
sales_monthly_sort_id['month'] = sales_monthly_sort_id['month'].astype('object')
sales_monthly_sort_id['year'] = sales_monthly_sort_id['year'].astype('object')
print(sales_monthly_sort_id.head(10))
print(sales_monthly_sort_id.shape)
sns.distplot(test.shop_id, bins=60, kde=False)

print(test.shop_id.nunique())
print(test.shop_id.unique())
test_new = pd.merge(test, items, on=['item_id'], how='left')
test_new['date_block_num'] = 34
test_new['year'] = '2015'
test_new['month'] = '11'
print(test_new.head())
X_test = test_new.drop(columns='ID').values
X = sales_monthly_sort_id.drop(columns=['ID', 'item_cnt_month']).values
y = sales_monthly_sort_id[['item_cnt_month']]
rf = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_leaf=10, criterion='mse', random_state=42, n_jobs=1)