import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 100)
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input2.head()
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input4.head()
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input3.head()
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input1.head()
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input5.head()
_input0 = _input0.drop(['date', 'item_price'], axis=1, inplace=False)
_input0.head()
_input0 = pd.merge(_input0, _input4.loc[:, ['item_id', 'item_category_id']], how='left', on='item_id')
_input0.head()
_input0 = _input0.groupby(['date_block_num', 'shop_id', 'item_id', 'item_category_id']).agg('sum')
_input0 = _input0.reset_index(inplace=False)
_input0 = pd.DataFrame(_input0)
_input0 = _input0.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=False)
_input0.head()
_input0.tail()
_input2 = pd.merge(_input2, _input4.loc[:, ['item_id', 'item_category_id']], how='left', on='item_id')
_input2.head()
_input2['date_block_num'] = 34
_input2.head()
_input2 = _input2.drop(['ID'], axis=1, inplace=False)
_input2.head()
all_data = pd.concat((_input0, _input2)).reset_index(drop=True)
all_data.head()
all_data.isna().sum().sort_values(ascending=False)
(_input0.shape, _input2.shape, all_data.shape)
_input0 = all_data.iloc[:1609124, :]
_input2 = all_data.iloc[-214200:, :]
(_input0.shape, _input2.shape)
_input2 = _input2.drop(['item_cnt_month'], axis=1, inplace=False)
x_train = _input0.drop('item_cnt_month', axis=1)
y_train = _input0[['item_cnt_month']]
(x_train.shape, y_train.shape)