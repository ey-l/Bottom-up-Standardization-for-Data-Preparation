import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('Sales train data info')
_input0.info()
_input0.head(5)
print('Test data info')
_input2.info()
_input2.head(5)
print('Sales Train data')
print('Null values:', _input0.isnull().values.any())
print('NaN values:', _input0.isna().values.any())
print()
print('Test data')
print('Null values:', _input2.isnull().values.any())
print('NaN values:', _input2.isna().values.any())
plt.figure(figsize=(10, 4))
sns.scatterplot(x=_input0.item_cnt_day, y=_input0.item_price, data=_input0)
_input0 = _input0[_input0.item_price < 75000]
_input0 = _input0[_input0.item_cnt_day < 1001]
_input0 = _input0[_input0.item_cnt_day >= 0]
plt.figure(figsize=(10, 4))
sns.scatterplot(x=_input0.item_cnt_day, y=_input0.item_price, data=_input0)
grouped_train = _input0.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].agg('sum').reset_index()
x = grouped_train.iloc[:, :-1]
y = grouped_train.iloc[:, -1:]
y = y.clip(0, 20)
print('X:')
print(x.head(5))
print('Y:')
print(y.head(5))
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=0)
model = RandomForestRegressor(n_estimators=25, random_state=0)