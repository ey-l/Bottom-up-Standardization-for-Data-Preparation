import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh as bk

from pylab import rcParams
rcParams['figure.figsize'] = (12, 5)
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.titlesize'] = 14
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
pd.options.display.max_columns = None
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', delimiter=',', engine='python', parse_dates=True)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', delimiter=',', engine='python')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv', delimiter=',', engine='python')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv', delimiter=',', engine='python')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv', delimiter=',', engine='python')
sales_train.head()
sales_train.info()
sales_train[['item_price', 'item_cnt_day']].describe()
sales_train.isnull().sum()
len(sales_train[sales_train.duplicated() == True])
sales_train[sales_train.duplicated() == True]
sales_train.drop_duplicates(keep='first', inplace=True)
test.head()
shops.head()
shops[shops.duplicated() == True]
item_categories.head()
item_categories[item_categories.duplicated() == True]
items[items.duplicated() == True]
items.head()
print(shops['shop_name'].nunique())
print(item_categories['item_category_id'].nunique())
print(items['item_id'].nunique())
sales_train.columns
items.columns
item_categories.columns
shops.columns
sales_train['shop_id'].nunique()
sales_train['item_id'].nunique()
sales_train.head(2)
sales_train.tail(3)
sales_train['shop_id'].nunique()
sales_train['shop_id'].value_counts().sort_values(ascending=False)
sales_train['item_id'].value_counts().sort_values(ascending=False)
sales_train.head(2)
sns.distplot(sales_train['item_price'])
plt.title('Histogram of the Item Price')

sns.distplot(sales_train['item_price'])
plt.xscale('log')
plt.title('Histogram of the Item Price (log scale)')

sns.boxplot(sales_train['item_price'], orient='h')
plt.title('Box Plot of the Item Price')
plt.grid()

sns.boxplot(sales_train['item_price'], orient='h')
plt.xscale('log')
plt.title('Box Plot of the Item Price (log scale)')

print(f'Median values of the item price is {np.median(sales_train.item_price)}')
(q3, q1) = np.percentile(sales_train['item_price'], [75, 25])
print(f'First Quartile of the Item Price {q1}')
print(f'Third Quartile of the Item Price {q3}')
IQR = q3 - q1
print(f'Interquartile Range of the Item Price {IQR}')
df_1 = sales_train[(sales_train['item_price'] < 1000.0) & (sales_train['item_price'] >= 249.0)]
sns.distplot(df_1['item_price'])
plt.title('Histogram of the Item Price')

sns.boxplot(df_1['item_price'], orient='h')
plt.title('Box Plot of the Item Price')

sns.kdeplot(df_1['item_cnt_day'])
plt.title('')

sales_train[sales_train['item_price'] > 300000]
np.min(sales_train['item_price'])
np.max(sales_train['item_price'])
sales_train[sales_train['item_price'] < 0]
sales_train['item_cnt_day'].describe()
np.min(sales_train['item_cnt_day'])
sales_train[sales_train['item_cnt_day'] < 0]['shop_id']
sales_train[sales_train['item_cnt_day'] < 0]['shop_id'].value_counts().sort_values(ascending=False).to_frame()[:10]
sales_train['item_cnt_day'].mask(sales_train['item_cnt_day'] < 0.0, 0.0, inplace=True)
sales_train['item_cnt_day'].describe()
sales_train.drop([sales_train[sales_train['item_price'] < 0].index[0]], inplace=True)
sales_train[sales_train['item_price'] < 0]
sales_train.head(3)
test.head(3)
sales_train.tail()
Q1 = np.percentile(sales_train['item_price'], 25, interpolation='midpoint')
Q3 = np.percentile(sales_train['item_price'], 75, interpolation='midpoint')
IQR = Q3 - Q1
print('Old Shape: ', sales_train.shape)
upper = np.where(sales_train['item_price'] >= Q3 + 1.5 * IQR)
lower = np.where(sales_train['item_price'] <= Q1 - 1.5 * IQR)
sales_train.drop(upper[0], inplace=True)
sales_train.drop(lower[0], inplace=True)
print('New Shape: ', sales_train.shape)
test['date_block_num'] = 34
test.head(2)
data_concat = pd.concat([sales_train, test])
data_concat.head()
data_concat.info()
data_concat.isnull().sum()
len(test)
data_concat.drop(['ID', 'date'], axis=1, inplace=True)
data_concat.head()
data = data_concat.groupby(by=['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].apply(sum)
data.head()
data.info()
data.isnull().sum()
data[data['date_block_num'] == 34].tail()
data['shop_lag_1'] = data.groupby('shop_id')['item_cnt_day'].shift(1)
data['shop_lag_2'] = data.groupby('shop_id')['item_cnt_day'].shift(2)
data['item_lag_1'] = data.groupby('item_id')['item_cnt_day'].shift(1)
data['item_lag_2'] = data.groupby('item_id')['item_cnt_day'].shift(2)
data[data['shop_id'] == 2]['item_id'].value_counts().sort_values(ascending=False)[:20]
data['shop_median'] = data.groupby(['shop_id'])['item_cnt_day'].median()
data['shop_mean'] = data.groupby(['shop_id'])['item_cnt_day'].mean()
data['item_median'] = data.groupby(['item_id'])['item_cnt_day'].median()
data['item_mean'] = data.groupby(['item_id'])['item_cnt_day'].mean()
data.head()
data.describe().transpose()
data.fillna(0.0, inplace=True)
data.isna().sum()
test_data = data[data['date_block_num'] == 34]
data_new = data[data['date_block_num'] != 34]
split_ratio = 0.8
train_data = data_new[int(split_ratio * len(data_new)):]
valid_data = data_new[len(train_data):]
(train_data.shape, test_data.shape, valid_data.shape)
X_train = train_data.drop('item_cnt_day', axis=1)
y_train = train_data['item_cnt_day']
X_valid = valid_data.drop('item_cnt_day', axis=1)
y_valid = valid_data['item_cnt_day']
X_test = test_data.drop('item_cnt_day', axis=1)
y_test = test_data['item_cnt_day']
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def test_submission(data, model):
    """prediction on the test data and generate the submission file"""
    predictions = model.predict(data)
    submission = test['ID'].to_frame()
    submission['item_cnt_month'] = predictions
    submission.head(3)

preprocess = Pipeline([('scaler', StandardScaler()), ('poly_features', PolynomialFeatures(degree=2)), ('decompose', PCA(n_components=0.9))])
X_train = preprocess.fit_transform(X_train)
X_valid = preprocess.transform(X_valid)
X_test = preprocess.transform(X_test)
lr_reg = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42, max_depth=5)
gb_reg = GradientBoostingRegressor(random_state=42)

def modeling(model, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test):
    """fit on the train data, print evaluation metrics and predict on the valid and test set"""