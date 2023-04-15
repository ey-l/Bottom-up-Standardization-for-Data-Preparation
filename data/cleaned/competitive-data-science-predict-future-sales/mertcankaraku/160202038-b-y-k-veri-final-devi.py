import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

def hazirla_sales_train(veri):
    veri = veri.groupby(['date_block_num', 'item_id', 'shop_id'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'max'})
    veri = veri[veri.item_price < 60000]
    veri = veri[veri.item_cnt_day < 12500]
    veri = veri.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'max_item_price'})
    return veri
egitim_verileri = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_verileri = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
test_verileri = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
ciktilar = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('test verileri :\n', test_verileri.head(20))
egitim_verileri = hazirla_sales_train(egitim_verileri)
item_verileri = item_verileri.drop(['item_name'], axis=1)
allTrainData = pd.merge(egitim_verileri, item_verileri)
allTestData = pd.merge(test_verileri, item_verileri)
allTestData = allTestData.drop(['ID'], axis=1)
allTestData['date_block_num'] = 34
df1 = allTrainData[['max_item_price', 'item_id', 'shop_id']]
allTestData = pd.merge(df1, allTestData)
(x_train, x_test, y_train, y_test) = train_test_split(allTrainData.drop('item_cnt_month', axis=1), allTrainData.item_cnt_month, test_size=0.33, random_state=0)
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(n_estimators=10, random_state=0)