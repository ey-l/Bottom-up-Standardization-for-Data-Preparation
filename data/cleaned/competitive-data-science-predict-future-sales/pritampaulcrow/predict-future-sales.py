import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train.shape
train.head()
train.info()
items.shape
items.head()
categories.shape
categories.head()
shops.shape
shops.head()
train.date_block_num.unique()
train.drop('date', axis=1, inplace=True)
train.corr()
sns.heatmap(train.corr())

def hist(data, x, bins, title, xlabel):
    plt.figure(figsize=(15, 12))
    sns.set()
    sns.distplot(data[x], color='blue')
    plt.title(title)
    plt.xlabel(xlabel)

hist(train, 'date_block_num', 1, 'date_block_num_dp', 'date_block_num')
hist(train, 'shop_id', 1, 'shop_id_dp', 'shop_id')
hist(train, 'item_id', 1, 'item_id_dp', 'item_id')
hist(train, 'item_price', 1000, 'item_price_dp', 'item_price')
plt.figure(figsize=(12, 8))
sns.boxplot(y=train['item_price'])
plt.ylim(0, 10000)
plt.grid()
plt.title('Boxplot for Item_price')
plt.ylabel('item_price')
test.head()
train.drop('date_block_num', axis=1, inplace=True)
x_train = train[['shop_id', 'item_id']]
y_train = train['item_cnt_day']
x_test = test[['shop_id', 'item_id']]
from sklearn.model_selection import train_test_split
(x_train1, x_val, y_train1, y_val) = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
modlr = LinearRegression()