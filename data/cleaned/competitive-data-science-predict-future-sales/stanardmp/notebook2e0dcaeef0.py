import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sale_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
datas = [items, item_categories, sale_train, shops, test]

def look(data):
    print(data.head())
    print('**************************')
    print(data.shape)
for d in datas:
    print('\n {} \n'.format(d))
    look(d)
d1 = sale_train.merge(shops, on='shop_id')
d2 = items.merge(item_categories, on='item_category_id')
df = d1.merge(d2, on='item_id')
df.head()
df.shape
df.info()
df.isnull().sum()
df1 = df.drop(['shop_name', 'item_category_name', 'item_name'], axis='columns')
df1.head()

def convert_date(data):
    data['day'] = pd.DatetimeIndex(pd.to_datetime(data['date'], format='%d.%m.%Y')).day
    data['month'] = pd.DatetimeIndex(pd.to_datetime(data['date'], format='%d.%m.%Y')).month
    data['year'] = pd.DatetimeIndex(pd.to_datetime(data['date'], format='%d.%m.%Y')).year
    return data
df2 = convert_date(df1)
df2.head()
df3 = df2.drop(['date'], axis='columns')
df3.head()
df3.duplicated().value_counts()
df4 = df3.drop_duplicates(subset=None, keep='first', inplace=False)
df4.duplicated().value_counts()
df4.shape
import seaborn as sns
import matplotlib.pyplot as plt
data = df4.copy()
plt.figure(figsize=(20, 5))
sns.countplot(x='month', data=data)
plt.title('Count of Sales each month')

plt.figure(figsize=(20, 5))
sns.countplot(x='year', data=data, palette='husl')
plt.title('Count of Sales each year')

years = data['year'].unique().tolist()
for y in years:
    d = data[data['year'] == y]
    print('*** Year {} ****\n'.format(y))
    df = d[['month', 'item_cnt_day']].groupby(['month']).sum().reset_index()
    plt.figure(figsize=(20, 5))
    sns.countplot(x='month', data=d)
    plt.title('Year {} sale per months'.format(y))

data.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
data.head()
data = data.drop(['day'], axis='columns')
data.head()
data = data.drop(['item_category_id'], axis='columns')
data.head()
data.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
y = data['item_cnt_month']
features = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'month', 'year']
x = data[features]
(X_train, X_val, Y_train, Y_val) = train_test_split(x, y, test_size=0.2, random_state=1)
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rfr = RandomForestRegressor(n_estimators=50, random_state=2)