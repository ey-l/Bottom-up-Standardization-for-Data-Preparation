import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, metrics
parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], date_parser=parser)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('train:', train.shape, 'test:', test.shape, 'items:', items.shape, 'item_cats:', item_cats.shape, 'shops:', shops.shape)
test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()
print('test only items:', len(test_only))
subset = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)
test_shops = test.shop_id.unique()
test_items = test.item_id.unique()
train = train[train.shop_id.isin(test_shops)]
train = train[train.item_id.isin(test_items)]
print('train:', train.shape)
from itertools import product
block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num', 'shop_id'])
shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id', 'item_id'])
all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')
print(len(all_combi), 34 * len(test_shops) * len(test_items))
train_base = pd.merge(all_combi, train, on=['date_block_num', 'shop_id', 'item_id'], how='left')
train_base['item_cnt_day'].fillna(0, inplace=True)
train_grp = train_base.groupby(['date_block_num', 'shop_id', 'item_id'])
train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day': ['sum', 'count']})).reset_index()
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt', 'item_order']
print(train_monthly[['item_cnt', 'item_order']].describe())
train_monthly['item_cnt'].clip(0, 20, inplace=True)
train_monthly.head()
item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])
item_grp = pd.Categorical(item_grp).codes
item_cats['item_group'] = item_grp
items = pd.merge(items, item_cats.loc[:, ['item_category_id', 'item_group']], on=['item_category_id'], how='left')
city = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(city).codes
grp = train_monthly.groupby(['shop_id', 'item_id'])
train_shop = grp.agg({'item_cnt': ['mean', 'median', 'std'], 'item_order': 'mean'}).reset_index()
train_shop.columns = ['shop_id', 'item_id', 'cnt_mean_shop', 'cnt_med_shop', 'cnt_std_shop', 'order_mean_shop']
print(train_shop[['cnt_mean_shop', 'cnt_med_shop', 'cnt_std_shop']].describe())
train_shop.head()
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')
grp = train_cat_monthly.groupby(['shop_id', 'item_group'])
train_shop_cat = grp.agg({'item_cnt': ['mean']}).reset_index()
train_shop_cat.columns = ['shop_id', 'item_group', 'cnt_mean_shop_cat']
print(train_shop_cat.loc[:, ['cnt_mean_shop_cat']].describe())
train_shop_cat.head()
train_prev = train_monthly.copy()
train_prev['date_block_num'] = train_prev['date_block_num'] + 1
train_prev.columns = ['date_block_num', 'shop_id', 'item_id', 'cnt_prev', 'order_prev']
for i in [2, 12]:
    train_prev_n = train_monthly.copy()
    train_prev_n['date_block_num'] = train_prev_n['date_block_num'] + i
    train_prev_n.columns = ['date_block_num', 'shop_id', 'item_id', 'cnt_prev' + str(i), 'order_prev' + str(i)]
    train_prev = pd.merge(train_prev, train_prev_n, on=['date_block_num', 'shop_id', 'item_id'], how='left')
train_prev.head()
grp = pd.merge(train_prev, items, on=['item_id'], how='left').groupby(['date_block_num', 'shop_id', 'item_group'])
train_cat_prev = grp['cnt_prev'].mean().reset_index()
train_cat_prev = train_cat_prev.rename(columns={'cnt_prev': 'cnt_prev_cat'})
print(train_cat_prev.loc[:, ['cnt_prev_cat']].describe())
train_cat_prev.head()
train_piv = train_monthly.pivot_table(index=['shop_id', 'item_id'], columns=['date_block_num'], values='item_cnt', aggfunc=np.sum, fill_value=0)
train_piv = train_piv.reset_index()
train_piv.head()
col = np.arange(34)
pivT = train_piv[col].T
ema_s = pivT.ewm(span=12).mean().T
ema_l = pivT.ewm(span=26).mean().T
macd = ema_s - ema_l
sig = macd.ewm(span=9).mean()
ema_list = []
for c in col:
    sub_ema = pd.concat([train_piv.loc[:, ['shop_id', 'item_id']], pd.DataFrame(ema_s.loc[:, c]).rename(columns={c: 'cnt_ema_s_prev'}), pd.DataFrame(ema_l.loc[:, c]).rename(columns={c: 'cnt_ema_l_prev'}), pd.DataFrame(macd.loc[:, c]).rename(columns={c: 'cnt_macd_prev'}), pd.DataFrame(sig.loc[:, c]).rename(columns={c: 'cnt_sig_prev'})], axis=1)
    sub_ema['date_block_num'] = c + 1
    ema_list.append(sub_ema)
train_ema_prev = pd.concat(ema_list)
train_ema_prev.head()
(fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
train_monthly.groupby(['date_block_num']).sum().reset_index()['item_cnt'].plot(ax=ax[0])
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')
train_cat_monthly.pivot_table(index=['date_block_num'], columns=['item_group'], values='item_cnt', aggfunc=np.sum, fill_value=0).plot(ax=ax[1], legend=False)
train_price = train_grp['item_price'].mean().reset_index()
price = train_price[~train_price['item_price'].isnull()]
last_price = price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last').drop(['date_block_num'], axis=1)
"\nmean_price = price.groupby(['item_id'])['item_price'].mean().reset_index()\nresult_price = pd.merge(test, mean_price, on=['item_id'], how='left').drop('ID', axis=1)\npred_price_set = result_price[result_price['item_price'].isnull()]\n"
uitem = price['item_id'].unique()
pred_price_set = test[~test['item_id'].isin(uitem)].drop('ID', axis=1)
if len(pred_price_set) > 0:
    train_price_set = pd.merge(price, items, on=['item_id'], how='inner')
    pred_price_set = pd.merge(pred_price_set, items, on=['item_id'], how='inner').drop(['item_name'], axis=1)
    reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)