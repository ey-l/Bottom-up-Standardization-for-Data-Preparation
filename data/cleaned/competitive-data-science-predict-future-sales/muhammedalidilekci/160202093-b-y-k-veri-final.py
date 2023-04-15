import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
dataST = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
itemsData = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
dataTest = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('dataST verisinin özeti')
print('------------------------------------------------------------')
print(dataST.head(10))
dataST = dataST.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'max'})
dataST = dataST[dataST.item_price < 40000]
dataST = dataST[dataST.item_cnt_day < 7500]
dataST = dataST.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'max_item_price'})
print('dataST verisinin  işlenmiş halinin özeti')
print('------------------------------------------------------------')
print(dataST.head(10))
itemsData = itemsData.drop(['item_name'], axis=1)
allTrainData = pd.merge(dataST, itemsData)
allTestData = pd.merge(dataTest, itemsData)
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