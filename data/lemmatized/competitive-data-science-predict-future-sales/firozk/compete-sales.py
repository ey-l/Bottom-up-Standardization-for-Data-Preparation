import pandas as pd
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0 = _input0.drop(columns=['date'])
_input0 = _input0.drop(columns=['date_block_num'])
_input0.head()
X = _input0.iloc[:, :3]
y = _input0.iloc[:, 3:]
from sklearn.preprocessing import Normalizer