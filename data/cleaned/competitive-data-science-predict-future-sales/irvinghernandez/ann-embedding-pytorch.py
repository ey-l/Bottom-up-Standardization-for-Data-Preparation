import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df = pd.merge(df, items[['item_id', 'item_category_id']], how='left', on='item_id')
df.sort_values(by='date', ascending=True, inplace=True)
df['month'] = df['date'].dt.month
df.head(5)
cat = ['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'month']
cont = ['item_price']
output = 'item_cnt_day'
df_agg = df.groupby(cat).agg({cont[0]: np.mean, output: np.sum}).reset_index()
del df
df_agg.info()
list_cluster = ['shop_id', 'item_id', 'item_category_id', 'item_price']
SSE = []
for cluster in range(1, 20):
    kmeans = MiniBatchKMeans(n_clusters=cluster, init='k-means++')