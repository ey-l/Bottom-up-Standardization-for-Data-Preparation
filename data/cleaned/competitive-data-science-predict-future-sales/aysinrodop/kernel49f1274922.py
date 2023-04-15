import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
from numpy.random import seed
from numpy.random import randn
import pandas as pd
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import seaborn as sns
sns.set(style='whitegrid')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_train.head(10)
sales_train.info()
sales_train[sales_train.isnull().any(axis=1)].sum()
sales_test[sales_test.isnull().any(axis=1)].sum()
plt.plot(sales_train['item_id'], sales_train['item_price'], 'o', color='blue')
nf = sales_train.select_dtypes(include=[np.number])
nf.dtypes
corr = nf.corr()
corr
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=0.8, square=True)
sales_train = sales_train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
sales_train.head(5)
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
sales_train['month'] = sales_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
sales_train['year'] = sales_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
sales_train = sales_train.drop('date', axis=1)
sales_train = sales_train.drop('item_category_id', axis=1)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
sales_train.groupby(['item_id', 'date_block_num', 'shop_id']).mean()
(X_train, Y_train) = (sales_train.drop(['month', 'year', 'item_price', 'item_cnt_day'], axis=1), sales_train.item_cnt_day)
X_train
X_train = X_train.values
X_train
Y_train = Y_train.values
Y_train
tahmin = sales_test
sales_test = sales_test.values
sales_test
model = LinearRegression()