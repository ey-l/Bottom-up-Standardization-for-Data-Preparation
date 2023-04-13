import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from time import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.options.mode.chained_assignment = None
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info('test')
pd.pandas.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(_input1.shape)
_input1.head()
_input1 = _input1.loc[:, _input1.isnull().sum() / len(_input1) < 0.8]
x = _input1.iloc[:, 1:-1]
y = _input1.iloc[:, -1]
train_cols = x.columns
print(_input1.shape, x.shape, y.shape)
train_stats = x.describe().transpose()
train_stats
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
ordinal_cols = list(x.columns[x.columns.str.contains('Yr|Year')])
print('ordinal/temporal columns:\n', ordinal_cols)
nominal_cols = list(set(x.select_dtypes(include=['object']).columns) - set(ordinal_cols))
print('nominal columns:\n', nominal_cols)
numeric_cols = list(set(x.select_dtypes(exclude=['object']).columns) - set(ordinal_cols))
print('numeric columns:\n', numeric_cols)
x[nominal_cols].describe().transpose()
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def missing_val_imputation(x, ordinal_cols, nominal_cols, numeric_cols):
    for col in ordinal_cols:
        x.loc[:, col] = x.loc[:, col].fillna(x.loc[:, col].mean())
    x.loc[:, nominal_cols] = x.loc[:, nominal_cols].fillna('?')
    for col in numeric_cols:
        x.loc[:, col] = x.loc[:, col].fillna(x.loc[:, col].mean())
    print('All missing values are now imputed!\n', x.isnull().sum().sort_values(ascending=False))
    return x
x_train = missing_val_imputation(x_train, ordinal_cols, nominal_cols, numeric_cols)
x_test = missing_val_imputation(x_test, ordinal_cols, nominal_cols, numeric_cols)