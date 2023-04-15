import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('Sales train data info')
sales_train.info()
sales_train.head(5)
print('Test data info')
test.info()
test.head(5)
print('Sales Train data')
print('Null values:', sales_train.isnull().values.any())
print('NaN values:', sales_train.isna().values.any())
print()
print('Test data')
print('Null values:', test.isnull().values.any())
print('NaN values:', test.isna().values.any())
plt.figure(figsize=(10, 4))
sns.scatterplot(x=sales_train.item_cnt_day, y=sales_train.item_price, data=sales_train)
sales_train = sales_train[sales_train.item_price < 75000]
sales_train = sales_train[sales_train.item_cnt_day < 1001]
sales_train = sales_train[sales_train.item_cnt_day >= 0]
plt.figure(figsize=(10, 4))
sns.scatterplot(x=sales_train.item_cnt_day, y=sales_train.item_price, data=sales_train)
grouped_train = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].agg('sum').reset_index()
x = grouped_train.iloc[:, :-1]
y = grouped_train.iloc[:, -1:]
y = y.clip(0, 20)
print('X:')
print(x.head(5))
print('Y:')
print(y.head(5))
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=0)
model = RandomForestRegressor(n_estimators=25, random_state=0)