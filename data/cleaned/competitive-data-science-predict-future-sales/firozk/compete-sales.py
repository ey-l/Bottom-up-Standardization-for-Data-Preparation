import pandas as pd
data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
data = data.drop(columns=['date'])
data = data.drop(columns=['date_block_num'])
data.head()
X = data.iloc[:, :3]
y = data.iloc[:, 3:]
from sklearn.preprocessing import Normalizer