import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import math
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy import stats
from scipy.stats import norm, skew
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
print('The train data size before dropping Id feature is : {} '.format(train_df.shape))
print('The test data size before dropping Id feature is : {} '.format(test_df.shape))
train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop('Id', axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)
print('\nThe train data size after dropping Id feature is : {} '.format(train_df.shape))
print('The test data size after dropping Id feature is : {} '.format(test_df.shape))
train_df.head()
test_df.head()
(fig, ax) = plt.subplots()
ax.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
(fig, ax) = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

train_df['SalePrice'].describe()
sns.distplot(train_df['SalePrice'], fit=norm)