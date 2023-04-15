import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sales_train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('Number of Training Samples = {}'.format(sales_train_df.shape[0]))
print('Number of Test Samples = {}\n'.format(test_df.shape[0]))
print('Training X Shape = {}'.format(sales_train_df.shape))
print('Test X Shape = {}'.format(test_df.shape))
print('Test y Shape = {}\n'.format(test_df.shape[0]))
print('Index of Train set:\n', sales_train_df.columns)
print(sales_train_df.info())
print('\nIndex of Test set:\n', test_df.columns)
print('\nMissing values of Train set:\n', sales_train_df.isnull().sum())
print('\nNull values of Train set:\n', sales_train_df.isna().sum())
sales_train_df.sample(10)
test_df.head()

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
sales_train_df = downcast_dtypes(sales_train_df)
print(sales_train_df.info())
sales_train_df['date'] = pd.to_datetime(sales_train_df['date'], format='%d.%m.%Y')
print('Min date from train set: %s' % sales_train_df['date'].min().date())
print('Max date from train set: %s' % sales_train_df['date'].max().date())
print('Min date_block_num from train set: %s' % sales_train_df['date_block_num'].min())
print('Max date_block_num from train set: %s' % sales_train_df['date_block_num'].max())
ts = sales_train_df.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(15, 8))
plt.title('Total Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sb.boxplot(x=sales_train_df['item_cnt_day'])
print('Sale volume outliers:', sales_train_df['item_id'][sales_train_df['item_cnt_day'] > 900].unique())
plt.figure(figsize=(10, 4))
plt.xlim(sales_train_df['item_price'].min(), sales_train_df['item_price'].max())
sb.boxplot(x=sales_train_df['item_price'])
print('Item price outliers:', sales_train_df['item_id'][sales_train_df['item_price'] > 300000].unique())
sales_train_df = sales_train_df[(sales_train_df.item_price < 300000) & (sales_train_df.item_cnt_day < 900)]
train_data = sales_train_df.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
train_data.reset_index(inplace=True)
train_data.head()
all_data = pd.merge(test_df, train_data, on=['item_id', 'shop_id'], how='left')
all_data.fillna(0, inplace=True)
all_data.head()
all_data.drop(['ID', 'shop_id', 'item_id'], inplace=True, axis=1)
all_data.head()

def split_sequences(sequences, n_steps):
    (X, y) = (list(), list())
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        (seq_x, seq_y) = (sequences[i:end_ix, :-1], sequences[end_ix - 1, -1])
        X.append(seq_x)
        y.append(seq_y)
    return (array(X), array(y))
dummy_samples = [[0] * 34] * 2
all_data.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
all_data = pd.concat([pd.DataFrame(dummy_samples), all_data], ignore_index=True)
all_data.shape
train_data = np.expand_dims(all_data.values[:, :-2], axis=2)
validation_data = np.expand_dims(all_data.values[:, 1:-1], axis=2)
test_data = np.expand_dims(all_data.values[:, 2:], axis=2)
print(train_data.shape, validation_data.shape, test_data.shape)
from numpy import array
n_steps = 3
(X_train, y) = split_sequences(train_data, n_steps)
(X_val, y_val) = split_sequences(validation_data, n_steps)
(X_test, y_test) = split_sequences(test_data, n_steps)
print(X_train.shape, y.shape)
X = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_val = X_val.reshape((X_val.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2]))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow.keras.optimizers as optimizers
n_features = X.shape[2]
model = Sequential()
model.add(LSTM(64, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(64, activation='relu', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse', metrics=['mean_squared_error'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [EarlyStopping(patience=5, verbose=1), ReduceLROnPlateau(factor=0.25, patience=2, min_lr=1e-06, verbose=1), ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)]