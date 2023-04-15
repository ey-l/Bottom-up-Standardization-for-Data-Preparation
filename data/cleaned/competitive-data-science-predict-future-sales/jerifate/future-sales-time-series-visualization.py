import re
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
item_categories_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
print('item_categories_df Shape : {}'.format(item_categories_df.shape))
item_categories_df.head()
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops_df.head()
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
print('items_df Shape : {}'.format(items_df.shape))
items_df.head()
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
print('train dataset Shape : {}'.format(train_df.shape))
train_df.head()
print('train dataset Shape : {}'.format(train_df.shape))
print('test dataset Shape : {}'.format(test_df.shape))
train_df.info()
train_df.describe()
sns.color_palette('YlOrRd', as_cmap=True)
Reds_palette = sns.color_palette('Reds', 10)
YlOrBr_palette = sns.color_palette('YlOrRd', 10)
sns.palplot(Reds_palette)
sns.palplot(YlOrBr_palette)
train_df.head()
pd.DataFrame(train_df.iloc[:, 4].sort_values(ascending=False)).head(10).style.background_gradient(cmap='Reds')
pd.DataFrame(train_df.iloc[:, 5].sort_values(ascending=False)).head(10).style.background_gradient(cmap='YlOrBr')
(fig, axes) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
sns.boxplot(train_df['item_price'], ax=axes[0])
axes[0].set_title('Distribution [item_price] Boxplots', fontweight='bold', fontfamily='serif', fontsize=14)
axes[0].patch.set_alpha(0)
sns.boxplot(train_df['item_cnt_day'], ax=axes[1])
axes[1].set_title('Distribution [item_cnt_day] Boxplots', fontweight='bold', fontfamily='serif', fontsize=14)
axes[1].patch.set_alpha(0)


def preporcess_data(df):
    print('Before cleansing shape : {}'.format(df.shape))
    print('----- CLEANSING START -----')
    df = df.drop(df[df['item_price'] >= 45000].index)
    df = df.drop(df[df['item_price'] < 0].index)
    df = df.drop(df[df['item_cnt_day'] >= 600].index)
    df = df.drop(df[df['item_cnt_day'] < 0].index)
    print(df.shape)
    print('----- CLEANSING END -----')
    print('After cleansing shape : {}'.format(df.shape))
    return df
train_df = preporcess_data(train_df)
Reds_palette_59 = sns.color_palette('Reds', 59)
(fig, ax) = plt.subplots(1, 1, figsize=(15, 6))
sns.countplot(data=train_df, x='shop_id', ax=ax, palette=Reds_palette_59)
ax.set_title('Destribute "Shop ID" count of train data', fontweight='bold', fontfamily='serif', fontsize=18)
ax.set(xlabel='Shop ID', ylabel='')
ax.patch.set_alpha(0)

YlOrBr_palette_89 = sns.color_palette('YlOrRd', 89)

def input_items_key_output_items_value(key):
    return item_dict[key]
item_dict = {key: value for (key, value) in zip(items_df['item_id'], items_df['item_category_id'])}
print('item_dict size : {}'.format(len(item_dict)))
train_df['item_category_id'] = train_df.apply(lambda x: input_items_key_output_items_value(x['item_id']), axis=1)
train_df.head()
(fig, ax) = plt.subplots(1, 1, figsize=(20, 7))
sns.countplot(data=train_df, x='item_category_id', ax=ax, palette=YlOrBr_palette_89)
ax.set_title('Destribute "Item Categori" count of train data', fontweight='bold', fontfamily='serif', fontsize=19)
ax.set(xlabel='Item Category ID', ylabel='')
ax.patch.set_alpha(0)

train_df['sales'] = train_df['item_price'] * train_df['item_cnt_day']
train_df.head()

def input_shopid_output_sales(tim_data, shop_id):
    shop_sales = []
    for i in range(len(tim_data)):
        a = train_df[(train_df['date_block_num'] == i) & (train_df['shop_id'] == shop_id)]['sales'].sum()
        shop_sales.append(a)
    return shop_sales
time_data = train_df['date_block_num'].unique()
(fig, axes) = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
x_idx = 0
y_idx = 0
for i in range(2 * 4):
    if x_idx == 4:
        x_idx = 0
        y_idx += 1
    random_index = np.random.randint(0, 61)
    sales_data = input_shopid_output_sales(time_data, random_index)
    sales_data_mean = np.mean(sales_data)
    axes[y_idx][x_idx].plot(sales_data, linewidth=4.0, color=Reds_palette[-1 * (i + 1)], label='Shop id {}'.format(random_index))
    axes[y_idx][x_idx].axhline(y=sales_data_mean, color='k', linestyle='--', label='Sales mean')
    axes[y_idx][x_idx].set_ylim(0, train_df['sales'].max() * 6)
    axes[y_idx][x_idx].set_title('Shop id : {} | mean : {:.2f}'.format(random_index, sales_data_mean), fontweight='bold', fontfamily='serif', fontsize=11)
    axes[y_idx][x_idx].legend()
    axes[y_idx][x_idx].set_xticks([])
    if x_idx != 0:
        axes[y_idx][x_idx].set_yticks([])
    axes[y_idx][x_idx].patch.set_alpha(0)
    x_idx += 1
fig.text(0, 1.08, 'Destribute sales figures for each store', fontweight='bold', fontfamily='serif', fontsize=18)

train_df.head()
group = train_df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
train_df = pd.merge(train_df, group, on=['shop_id', 'item_id', 'date_block_num'], how='left')
train_df['item_cnt_month'] = train_df['item_cnt_month'].fillna(0).clip(0, 100).astype(np.float16)
train_df[(train_df['shop_id'] == 31) & (train_df['date_block_num'] == 0) & (train_df['item_id'] == 4906)].head()
train_df = train_df.drop_duplicates(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month'])
print('train data Shape : {}'.format(train_df.shape))
train_df.head()
Reds_palette_89 = sns.color_palette('Reds', 89)
Reds_palette_100 = sns.color_palette('Reds', 100)
(fig, ax) = plt.subplots(1, 1, figsize=(12, 5))
sns.kdeplot(x='item_price', data=train_df, fill=True, cut=0, bw_method=0.2, color=Reds_palette[2], lw=1.4, ax=ax, alpha=0.3)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks([])
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax.patch.set_alpha(0)
ax.legend([], [], frameon=False)
fig.text(0.15, 0.91, 'Count distribution by Price in Data', fontweight='bold', fontfamily='serif', fontsize=17)

price_dict = {key: value for (key, value) in zip(train_df['item_id'], train_df['item_price'])}
print('price_dict size : {}'.format(len(price_dict)))
price_dict_df = pd.DataFrame.from_dict(price_dict, orient='index', columns=['price'])
price_df = price_dict_df.reset_index()
price_df.columns = ['item_id', 'price']
price_df.head()
price_list = []
for val in price_df['price'].values:
    price_list.append([val])
print(price_list[:10])
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
std_scaler = StandardScaler()