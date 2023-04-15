import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df = df[2885849:]
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test = test_data.drop('ID', axis=1)
df.head()
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import regex as re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
df = df.drop(columns=['date_block_num'], axis=1)
df['item_id'].fillna(df['item_id'].mean(), inplace=True)
df['item_id'].fillna(df['item_id'].mean(), inplace=True)
df['item_price'].fillna(df['item_price'].mean(), inplace=True)
df['item_cnt_day'].fillna(df['item_cnt_day'].mean(), inplace=True)

def add_datepart(df, fldname, drop=True, time=False):
    """Helper function that adds columns relevant to a date."""
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(fldname, axis=1, inplace=True)
add_datepart(df, 'date')
df = df.drop('Elapsed', axis=1)
df.head()
label_encoder = preprocessing.LabelEncoder()
df['Is_month_end'] = label_encoder.fit_transform(df['Is_month_end'])
df['Is_month_start'] = label_encoder.fit_transform(df['Is_month_start'])
df['Is_quarter_end'] = label_encoder.fit_transform(df['Is_quarter_end'])
df['Is_quarter_start'] = label_encoder.fit_transform(df['Is_quarter_start'])
df['Is_year_end'] = label_encoder.fit_transform(df['Is_year_end'])
df['Is_year_start'] = label_encoder.fit_transform(df['Is_year_start'])
df.head()
test['item_price'] = df['item_price']
test['item_price'].fillna(df['item_price'].mean(), inplace=True)
test['Year'] = df['Year']
test['Year'].fillna(df['Year'].mean(), inplace=True)
test['Month'] = df['Month']
test['Month'].fillna(df['Month'].mean(), inplace=True)
test['Week'] = df['Week']
test['Week'].fillna(df['Week'].mean(), inplace=True)
test['Day'] = df['Day']
test['Day'].fillna(df['Day'].mean(), inplace=True)
test['Dayofweek'] = df['Dayofweek']
test['Dayofweek'].fillna(df['Dayofweek'].mean(), inplace=True)
test['Dayofyear'] = df['Dayofyear']
test['Dayofyear'].fillna(df['Dayofyear'].mean(), inplace=True)
test['Is_month_end'] = df['Is_month_end']
test['Is_month_end'].fillna(df['Is_month_end'].mean(), inplace=True)
test['Is_month_start'] = df['Is_month_start']
test['Is_month_start'].fillna(df['Is_month_start'].mean(), inplace=True)
test['Is_quarter_end'] = df['Is_quarter_end']
test['Is_quarter_end'].fillna(df['Is_quarter_end'].mean(), inplace=True)
test['Is_quarter_start'] = df['Is_quarter_start']
test['Is_quarter_start'].fillna(df['Is_quarter_start'].mean(), inplace=True)
test['Is_year_end'] = df['Is_year_end']
test['Is_year_end'].fillna(df['Is_year_end'].mean(), inplace=True)
test['Is_year_start'] = df['Is_year_start']
test['Is_year_start'].fillna(df['Is_year_start'].mean(), inplace=True)
test.head()
item_cnt = df['item_cnt_day']
df = df.drop(labels='item_cnt_day', axis=1)
df.insert(15, 'item_cnt_day', item_cnt)
80 * len(df) / 100
train_df = df.loc[2885849:2925849]
valid_df = df.loc[2925849:]
(len(train_df), len(valid_df))
y_train = train_df['item_cnt_day']
x_train = train_df.drop('item_cnt_day', axis=1)
y_valid = valid_df['item_cnt_day']
x_valid = valid_df.drop('item_cnt_day', axis=1)
model1 = RandomForestClassifier()
model2 = XGBClassifier()
model3 = neighbors.KNeighborsClassifier()
model4 = SVC()
(x_train.shape, y_train.shape)
x_valid.shape