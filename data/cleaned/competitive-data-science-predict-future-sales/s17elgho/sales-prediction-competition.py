import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import time
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print('Shape of categories data:', cats.shape)
print('Shape of items data:', items.shape)
print('Shape of training set:', sales_train.shape)
print('Shape of shops data:', shops.shape)
print('Shape of test set:', test.shape)
sales_train.head()
print('Number of items is:', len(sales_train['item_id'].unique()), 'and number of shops is:', len(sales_train['shop_id'].unique()))
sales_train.describe()
print('-----------Information-----------')
print(sales_train.info())
plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sns.boxplot(x=sales_train.item_cnt_day)
sales_train = sales_train.loc[sales_train['item_cnt_day'] < 1000,]
print('Shape of training set:', sales_train.shape)
plt.figure(figsize=(10, 4))
plt.xlim(sales_train.item_price.min(), sales_train.item_price.max() * 1.1)
sns.boxplot(x=sales_train.item_price)
sales_train = sales_train.loc[sales_train['item_price'] < 90000,]
sales_train = sales_train.loc[sales_train['item_price'] > 0,]
print('Shape of training set:', sales_train.shape)
print('we deleted', 2935847 - 2935845, 'rows')
print('the number of missing values per variable\n', sales_train.isna().sum())
spec_train = sales_train.sort_values('date').groupby(['date_block_num', 'shop_id'], as_index=False)
spec_train = spec_train.agg({'item_cnt_day': ['sum']})
spec_train.columns = ['date_block_num', 'shop_id', 'item_month_by_shop']
spec_train.head()
axis = np.arange('2013-01', '2015-11', dtype='datetime64[M]')
plt.figure(figsize=(20, 15))
list_of_shops = sorted(spec_train['shop_id'].unique())
for i in range(len(list_of_shops)):
    L = spec_train.loc[spec_train['shop_id'] == list_of_shops[i], 'item_month_by_shop']
    if L.shape[0] == 34:
        plt.plot(axis, spec_train.loc[spec_train['shop_id'] == list_of_shops[i], 'item_month_by_shop'])
plt.title('sales per shop')

shops.head()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id', 'city_code']]
shops.head()
cats.head()
cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats.head()
cats = cats[['item_category_id', 'type_code', 'subtype_code']]
cats.head()
items.head()
items.drop(['item_name'], axis=1, inplace=True)
items.head()
final_train = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = sales_train[sales_train.date_block_num == i]
    final_train.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
final_train = pd.DataFrame(np.vstack(final_train), columns=cols)
final_train['date_block_num'] = final_train['date_block_num'].astype(np.int8)
final_train['shop_id'] = final_train['shop_id'].astype(np.int8)
final_train['item_id'] = final_train['item_id'].astype(np.int16)
final_train.sort_values(cols, inplace=True)
print(final_train.head(), '\n')
print('Shape of the final data is:', final_train.shape)
sales_train['year'] = sales_train['date'].apply(lambda x: x[6:])
sales_train['year'] = sales_train['year'].astype('int64')
dict_year = sales_train[['date_block_num', 'year']].set_index('date_block_num').to_dict()['year']
final_train['year'] = final_train['date_block_num'].map(dict_year)
final_train.head()
final_train['month'] = final_train['date_block_num'] % 12
final_train.head()
train_monthly = sales_train.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_cnt_day': ['sum']})
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']
train_monthly
final_train = pd.merge(final_train, train_monthly, on=cols, how='left')
final_train['item_cnt_month'] = final_train['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
final_train
final_train1 = final_train.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(cats, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
final_train1
group = final_train1.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)
final_train2 = pd.merge(final_train1, group, on=['date_block_num'], how='left')
final_train2['date_avg_item_cnt'] = final_train2['date_avg_item_cnt'].astype(np.float16)
final_train2.shape
group1 = final_train2[['date_block_num', 'shop_id', 'item_id', 'date_avg_item_cnt']].copy()
group1['date_avg_item_cnt_shift_1'] = group1['date_avg_item_cnt']
group1 = group1.drop(columns='date_avg_item_cnt')
group1['date_block_num'] += 1
final_train3 = pd.merge(final_train2, group1, on=['date_block_num', 'shop_id', 'item_id'], how='left').copy()
final_train3.shape
group2 = final_train3[['date_block_num', 'item_id', 'item_cnt_month']].groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group2.columns = ['item_cnt_month_mean_shift_1']
group2.reset_index(inplace=True)
group2['date_block_num'] += 1
final_train4 = pd.merge(final_train3, group2[['date_block_num', 'item_id', 'item_cnt_month_mean_shift_1']], on=['date_block_num', 'item_id'], how='left')
final_train4.shape
group3 = sales_train[['date_block_num', 'item_id', 'item_price']].groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
group3.columns = ['item_price_mean_shift_1']
group3.reset_index(inplace=True)
group3['date_block_num'] += 1
final_train5 = pd.merge(final_train4, group3[['date_block_num', 'item_id', 'item_price_mean_shift_1']], on=['date_block_num', 'item_id'], how='left')
final_train5.shape
group4 = final_train5.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group4.columns = ['date_shop_avg_item_cnt']
group4.reset_index(inplace=True)
final_train6 = pd.merge(final_train5, group4, on=['date_block_num', 'shop_id'], how='left')
final_train6['date_shop_avg_item_cnt'] = final_train6['date_shop_avg_item_cnt'].astype(np.float16)
final_train6.shape
group5 = final_train6.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group5.columns = ['date_shop_cat_avg_item_cnt']
group5.reset_index(inplace=True)
final_train7 = pd.merge(final_train6, group5, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
final_train7['date_shop_cat_avg_item_cnt'] = final_train7['date_shop_cat_avg_item_cnt'].astype(np.float16)
final_train7.shape
group6 = final_train7.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group6.columns = ['date_item_city_avg_item_cnt']
group6.reset_index(inplace=True)
final_train8 = pd.merge(final_train7, group6, on=['date_block_num', 'item_id', 'city_code'], how='left')
final_train8['date_item_city_avg_item_cnt'] = final_train8['date_item_city_avg_item_cnt'].astype(np.float16)
final_train8
corr = final_train8[['date_block_num', 'shop_id', 'item_id', 'year', 'month', 'item_cnt_month', 'item_category_id', 'city_code', 'type_code', 'subtype_code', 'date_avg_item_cnt', 'date_avg_item_cnt_shift_1', 'item_cnt_month_mean_shift_1', 'item_price_mean_shift_1', 'date_shop_avg_item_cnt', 'date_shop_cat_avg_item_cnt', 'date_item_city_avg_item_cnt']].corr(method='pearson')
cmap = sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector='th', props=[('font-size', '7pt')]), dict(selector='td', props=[('padding', '0em 0em')]), dict(selector='th:hover', props=[('font-size', '12pt')]), dict(selector='tr:hover td:hover', props=[('max-width', '200px'), ('font-size', '12pt')])]
corr.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '80px', 'font-size': '10pt'}).set_caption('Hover to magify').set_precision(2).set_table_styles(magnify())
test.head()
teest = test.drop(columns='ID')
teest['date_block_num'] = 34
teest['month'] = 11
teest['year'] = 2015
test1 = teest.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(cats, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
test1['date_avg_item_cnt_shift_1'] = 0.259033
test2 = pd.merge(test1, group2[['date_block_num', 'item_id', 'item_cnt_month_mean_shift_1']], on=['date_block_num', 'item_id'], how='left')
test3 = pd.merge(test2, group3[['date_block_num', 'item_id', 'item_price_mean_shift_1']], on=['date_block_num', 'item_id'], how='left')
test4 = pd.merge(test3, group4[['date_block_num', 'shop_id', 'date_shop_avg_item_cnt']], on=['date_block_num', 'shop_id'], how='left')
test5 = pd.merge(test4, group5[['date_block_num', 'shop_id', 'item_category_id', 'date_shop_cat_avg_item_cnt']], on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
test6 = pd.merge(test5, group6[['date_block_num', 'item_id', 'city_code', 'date_item_city_avg_item_cnt']], on=['date_block_num', 'item_id', 'city_code'], how='left')
test6
test6['item_cnt_month'] = -1
final_train9 = final_train8[['shop_id', 'item_id', 'date_block_num', 'month', 'year', 'item_category_id', 'city_code', 'type_code', 'subtype_code', 'date_avg_item_cnt_shift_1', 'item_cnt_month_mean_shift_1', 'item_price_mean_shift_1', 'date_shop_avg_item_cnt', 'date_shop_cat_avg_item_cnt', 'date_item_city_avg_item_cnt', 'item_cnt_month']]
train_test_set = pd.concat([final_train9, test6], axis=0)
train_test_set['item_shop_first_sale'] = train_test_set['date_block_num'] - train_test_set.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
train_test_set['item_first_sale'] = train_test_set['date_block_num'] - train_test_set.groupby('item_id')['date_block_num'].transform('min')
train_test_set
X_train = final_train8[final_train8.date_block_num < 26].drop(['item_cnt_month'], axis=1)[['shop_id', 'item_id', 'subtype_code', 'item_category_id', 'item_price_mean_shift_1', 'item_cnt_month_mean_shift_1']].fillna(0)
Y_train = final_train8[final_train8.date_block_num < 26]['item_cnt_month']
X_valid = final_train8[final_train8.date_block_num >= 26].drop(['item_cnt_month'], axis=1)[['shop_id', 'item_id', 'subtype_code', 'item_category_id', 'item_price_mean_shift_1', 'item_cnt_month_mean_shift_1']].fillna(0)
Y_valid = final_train8[final_train8.date_block_num >= 26]['item_cnt_month']
X_test = test6[['shop_id', 'item_id', 'subtype_code', 'item_category_id', 'item_price_mean_shift_1', 'item_cnt_month_mean_shift_1']].fillna(0)
rf = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)