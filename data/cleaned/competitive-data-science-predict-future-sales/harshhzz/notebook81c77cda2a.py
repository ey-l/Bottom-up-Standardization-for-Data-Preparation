from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df.head()
features = ['shop_id', 'item_id']
x = df[features]
y = df.item_cnt_day
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()), ('model', RandomForestRegressor(n_estimators=50, random_state=0))])
(train_x, val_x, train_y, val_y) = train_test_split(x, y, random_state=1, test_size=0.3, train_size=0.7)
from xgboost import XGBRegressor
my_model2 = XGBRegressor(learning_rate=0.25, n_estimators=600)