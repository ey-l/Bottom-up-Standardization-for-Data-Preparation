import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = _input1.copy()
test = _input0.copy()
train.head(10)
train.info()
train['MSSubClass'] = train['MSSubClass'].astype('object')
train = train.drop(['Id'], axis=1, inplace=False)
test['MSSubClass'] = test['MSSubClass'].astype('object')
test = test.drop(['Id'], axis=1, inplace=False)
train['MoSold'] = train['MoSold'].astype('object')
test['MoSold'] = test['MoSold'].astype('object')
train['YrSold'] = train['YrSold'].astype('object')
test['YrSold'] = test['YrSold'].astype('object')
cat_cols_train = []
cont_cols_train = []
for i in train.columns:
    if train[i].dtypes == 'object':
        cat_cols_train.append(i)
    else:
        cont_cols_train.append(i)
cat_cols_test = []
cont_cols_test = []
for i in test.columns:
    if test[i].dtypes == 'object':
        cat_cols_test.append(i)
    else:
        cont_cols_test.append(i)
sns.boxplot(train['SalePrice'])
from scipy.stats import norm