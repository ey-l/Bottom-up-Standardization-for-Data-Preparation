import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train.item_cnt_day.plot()
plt.title('Günlük satılan ürün sayısı')
target = train.item_cnt_day
train = train.drop(['item_price', 'item_cnt_day', 'date', 'date_block_num'], axis=1).select_dtypes(exclude=['object'])
(train_X, test_X, train_y, test_y) = train_test_split(train, target, test_size=0.2)
my_model = XGBRegressor(n_estimators=10, learning_rate=0.07)