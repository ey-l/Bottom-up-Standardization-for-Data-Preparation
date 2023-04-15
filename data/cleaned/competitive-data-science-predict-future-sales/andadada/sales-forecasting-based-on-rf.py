import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(df_test)
df_train.head()
df_test.head()
df_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
df_train.info()
print(df_train['shop_id'])
df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.strftime('%Y-%m'))
df_train.head()
X = df_train.groupby(['date', 'shop_id', 'item_id']).sum()
X = X.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
X.reset_index(inplace=True)
X.head()
X_t = pd.merge(df_test, X, on=['shop_id', 'item_id'], how='left')
X_t.drop(['ID'], axis=1, inplace=True)
u = X_t.select_dtypes(exclude=['datetime'])
X_t[u.columns] = u.fillna(0)
X_t.head()
Y = X['2015-10'].values
X_T = X.drop(['2015-10'], axis=1)
X_t = X_t.drop(['2015-10'], axis=1)
print(X_T.shape, Y.shape)
print(X_t.shape)
corrmat = X_t.corr()
sns.heatmap(corrmat, square=True)
k = 35
cols = corrmat.nlargest(k, '2015-09')['2015-09'].index
cm = np.corrcoef(X_t[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
plt.tick_params(labelsize=8)


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return (mae, mse, rmse, r2_square)
(x_train, x_val, y_train, y_val) = train_test_split(X_T, Y, test_size=0.2, random_state=3)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
RF_reg = RandomForestRegressor(n_estimators=100)