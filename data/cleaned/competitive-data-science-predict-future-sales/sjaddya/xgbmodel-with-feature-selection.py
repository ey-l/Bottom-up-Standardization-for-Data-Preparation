import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_category = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
"dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0, aggfunc='sum')\ndataset.reset_index(inplace = True)\ndataset.head()"
'test.head()'
"dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')\ndataset.fillna(0,inplace = True)\ndataset.head()"
"dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)\ndataset.head()"
'dataset.shape'
'# X we will keep all columns execpt the last one \nX_train = np.expand_dims(dataset.values[:,:-1],axis = 2)\n# the last column is our label\ny_train = dataset.values[:,-1:]\n\n# for test we keep all the columns execpt the first one\nX_test = np.expand_dims(dataset.values[:,1:],axis = 2)\n\n# lets have a look on the shape \nprint(X_train.shape,y_train.shape,X_test.shape)'
'from keras.models import Sequential\nfrom keras.layers import LSTM,Dense,Dropout'
"baseline_model = Sequential()\nbaseline_model.add(LSTM(units = 64,input_shape = (33,1)))\nbaseline_model.add(Dense(1))\n\nbaseline_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])\nbaseline_model.summary()"