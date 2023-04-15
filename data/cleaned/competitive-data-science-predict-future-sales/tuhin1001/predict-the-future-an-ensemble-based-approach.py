import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
warnings.filterwarnings('ignore')
data_path = '_data/input/competitive-data-science-predict-future-sales/'
pd.set_option('float_format', '{:f}'.format)
sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops = pd.read_csv(data_path + 'shops.csv')
items = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_train['year'] = sales_train['date'].dt.year
sales_train['month'] = sales_train['date'].dt.month
sales_train['day'] = sales_train['date'].dt.day
sales_train['day'] = 1
print(sales_train.describe())
sales_train = sales_train[sales_train['item_price'] > 0]
sales_train = sales_train[sales_train['item_price'] < 50000]
sales_train = sales_train[sales_train['item_cnt_day'] > 0]
sales_train = sales_train[sales_train['item_cnt_day'] < 1000]
data = sales_train.groupby(['shop_id']).agg({'item_id': 'nunique'}).reset_index()
mpl.rc('font', size=6)
(figure, ax) = plt.subplots()
figure.set_size_inches(11, 5)
data = data.reset_index()
sns.barplot(x='shop_id', y='item_id', data=data)
ax.set(title='Distribution of items sold across different shops', xlabel='Shop Number', ylabel='Total Items Sold')

print(shops['shop_name'][0], '||', shops['shop_name'][57])
print(shops['shop_name'][1], '||', shops['shop_name'][58])
print(shops['shop_name'][10], '||', shops['shop_name'][11])
print(shops['shop_name'][39], '||', shops['shop_name'][40])
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 39, 'shop_id'] = 40
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
label_encoder = LabelEncoder()
shops['city'] = label_encoder.fit_transform(shops['city'])
data3 = sales_train.groupby(['year', 'month', 'date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
data = pd.merge(data3, items, how='left', on='item_id')
data = pd.merge(data, item_categories, how='left', on='item_category_id')
data = pd.merge(data, shops, how='left', on='shop_id')
data['month'] = data['date_block_num'].apply(lambda month: (month + 1) % 12)
data = data[['month', 'date_block_num', 'shop_id', 'item_id', 'item_category_id', 'city', 'item_price', 'item_cnt_day']]
test['date_block_num'] = 34
test['month'] = 11
item_price = data[['item_id', 'item_price']].groupby('item_id')['item_price'].mean().reset_index()
test = pd.merge(test, item_price, how='left', on='item_id')
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_categories, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
target = ['item_cnt_day']
features = ['month', 'shop_id', 'item_id', 'item_category_id', 'city', 'item_price']
data1 = data[data['month'] < 10]
data2 = data[data['month'] == 10]
x_train = data1[features].fillna(value=0)
y_train = data1[target].fillna(value=0)
x_valid = data2[features].fillna(value=0)
y_valid = data2[target].fillna(value=0)
params = {'metric': 'rmse', 'num_leaves': 255, 'learning_rate': 0.005, 'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 5, 'force_col_wise': True, 'random_state': 10}
cat_features = features
dtrain = lgb.Dataset(x_train, y_train)
dvalid = lgb.Dataset(x_valid, y_valid)
lgb_model = lgb.train(params=params, train_set=dtrain, num_boost_round=1500, valid_sets=(dtrain, dvalid), early_stopping_rounds=150, categorical_feature=cat_features, verbose_eval=100)
test1 = test[features]
test1 = test1.fillna(0)
test['preds'] = lgb_model.predict(test1)
preds = test[['ID', 'preds']]
preds.columns = ['ID', 'item_cnt_month']
