import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from simple_colors import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_cat
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales
shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shop.head(2)
s = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
s
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test
test.isnull().sum()
test_fact = ['ID', 'shop_id', 'item_id']
x_test = test[test_fact]
dataFrame = sales.copy()
dataFrame = pd.merge(item_cat, item, how='inner', on='item_category_id')
dataFrame = pd.merge(dataFrame, sales, how='inner', on='item_id')
dataFrame = pd.merge(dataFrame, shop, how='inner', on='shop_id')
dataFrame
dataFrame.describe().T.sort_values(ascending=0, by='mean').style.background_gradient(cmap='BuGn').bar(subset=['std'], color='red').bar(subset=['mean'], color='blue')
dataFrame.describe(include='object')
dataFrame.columns
col = ['item_category_name', 'item_category_id', 'item_name', 'item_id', 'date', 'date_block_num', 'shop_id', 'item_price', 'item_cnt_day', 'shop_name']
for i in col:
    print(cyan(i.capitalize(), ['bold', 'underlined']))
    print(magenta('Number of Unique values:'), magenta(dataFrame[i].nunique(), ['bold']))
    print(blue('------------------------------'))
    print(red('Value Count of '), red(i.capitalize()))
    print(dataFrame[i].value_counts())
    print('')
    print(yellow('*************************************************************************************************', ['bold']))
    print('')
dataFrame['item_category_name'].value_counts().plot(kind='bar', color='orange', figsize=(20, 8))
dataFrame['item_category_id'].value_counts().plot(kind='bar', color='pink', figsize=(20, 8))
dataFrame['item_name'].value_counts().plot(kind='hist', color='red', figsize=(10, 8), bins=50)
dataFrame['item_id'].value_counts().plot(kind='hist', figsize=(10, 8))
dataFrame['date'].value_counts().plot(kind='hist', color='crimson', figsize=(20, 8))
dataFrame['date_block_num'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(20, 8))
dataFrame['shop_id'].value_counts().plot(kind='bar', color='blue', figsize=(20, 8))
dataFrame['item_price'].value_counts().plot(kind='hist', color='cyan', figsize=(20, 8), bins=50)
dataFrame['item_cnt_day'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(20, 8))
dataFrame['shop_name'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(20, 8))
dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%d.%m.%Y')
dataFrame['month'] = dataFrame['date'].dt.month
dataFrame = dataFrame[['month', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum', 'month': 'min'}).reset_index()
dataFrame.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
dataFrame = dataFrame.sort_values(by=['date_block_num'], ascending=True).reset_index(drop=True)
dataFrame
dataFrame.corr()
sns.heatmap(dataFrame.corr())
dataFrame.corr()['item_cnt_month'].sort_values()
dataFrame.isnull().sum()
x_train = ['shop_id', 'item_id']
x = dataFrame[x_train]
y = dataFrame['item_cnt_month']
print(x)
y
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
test_x = test[['shop_id', 'item_id']]
type(test_x)
model = LinearRegression()