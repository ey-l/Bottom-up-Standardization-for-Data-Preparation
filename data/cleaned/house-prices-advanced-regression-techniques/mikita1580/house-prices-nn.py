import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = train.append(test)
df.head()
df.describe()
missing = df.isnull().sum()[df.isnull().sum() > 0]
missing_df = pd.DataFrame({'NaN_count': missing, 'NaN_percentage': missing / len(df)}).sort_values(by='NaN_percentage', ascending=False)
missing_df.head(10)
df = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])
num_cols = [i for i in df.columns if df[i].dtype in ['int', 'float']]
cat_cols = [i for i in df.columns if df[i].dtype == 'object']
import matplotlib.pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(12, 3, figsize=(20, 50))
c = 0
for i in range(12):
    for j in range(3):
        if c == 36:
            break
        sns.histplot(x=df.reset_index(drop=True)[num_cols[c]], ax=ax[i, j])
        c += 1

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
num_cols = df.describe().columns.tolist()
num_cols.remove('Id')
df = pd.get_dummies(df, drop_first=True)
imp = KNNImputer(n_neighbors=10, weights='uniform')
ID = df.pop('Id')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
df['Id'] = ID.tolist()
df[num_cols] = np.log(df[num_cols] + 1)
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
for col in num_cols:
    df[col] = df_normalized[col]
train_prepared = df.loc[df['Id'].isin(train['Id'])]
test_prepared = df.loc[df['Id'].isin(test['Id'])]
from sklearn.model_selection import train_test_split
(tr, val) = train_test_split(train_prepared.drop('Id', axis=1), test_size=0.3, random_state=42)
(X_train, X_valid) = (tr.drop('SalePrice', axis=1), val.drop('SalePrice', axis=1))
(Y_train, Y_valid) = (tr.SalePrice, val.SalePrice)
X_train = X_train.values
X_valid = X_valid.values
Y_train = Y_train.values
Y_valid = Y_valid.values
import tensorflow as tf
from tensorflow import keras
np.random.seed(42)
tf.random.set_seed(42)
lstm_mult_model2 = keras.models.Sequential([keras.layers.Dense(150, activation='relu', input_shape=[1, X_train.shape[1]], kernel_initializer='random_normal', bias_initializer='zeros'), keras.layers.Dropout(rate=0.5), keras.layers.Dense(50, activation='relu'), keras.layers.Dropout(rate=0.5), keras.layers.Dense(1)])
optimizer = keras.optimizers.Adam(learning_rate=0.0015)
lstm_mult_model2.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError()])
lstm_mult_model2.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min', restore_best_weights=True)