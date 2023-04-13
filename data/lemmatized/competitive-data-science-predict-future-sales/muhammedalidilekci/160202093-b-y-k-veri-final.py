import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('dataST verisinin özeti')
print('------------------------------------------------------------')
print(_input0.head(10))
_input0 = _input0.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'max'})
_input0 = _input0[_input0.item_price < 40000]
_input0 = _input0[_input0.item_cnt_day < 7500]
_input0 = _input0.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'max_item_price'})
print('dataST verisinin  işlenmiş halinin özeti')
print('------------------------------------------------------------')
print(_input0.head(10))
_input4 = _input4.drop(['item_name'], axis=1)
allTrainData = pd.merge(_input0, _input4)
allTestData = pd.merge(_input2, _input4)
allTestData = allTestData.drop(['ID'], axis=1)
allTestData['date_block_num'] = 34
print('allTestData verisinin özeti')
print('------------------------------------------------------------')
print(allTestData.head(10))
df1 = allTrainData[['max_item_price', 'item_id', 'shop_id']]
allTestData = pd.merge(df1, allTestData)
print('allTestData verisinin özeti')
print('------------------------------------------------------------')
print(allTestData.head(10))
(x_train, x_test, y_train, y_test) = train_test_split(allTrainData.drop('item_cnt_month', axis=1), allTrainData.item_cnt_month, test_size=0.33, random_state=0)
print('x_train verisinin özeti')
print('------------------------------------------------------------')
print(x_train.head(10))
print('\n\ny_train verisinin özeti')
print('------------------------------------------------------------')
print(y_train.head(10))
reg_decT = DecisionTreeRegressor(random_state=0)