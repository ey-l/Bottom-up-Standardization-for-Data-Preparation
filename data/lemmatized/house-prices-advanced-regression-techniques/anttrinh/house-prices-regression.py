import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy import stats
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
from subprocess import check_output
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.T
_input0.head(5)
print('The train data size before dropping Id feature is : {} '.format(_input1.shape))
print('The test data size before dropping Id feature is : {} '.format(_input0.shape))
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
print('\nThe train data size after dropping Id feature is : {} '.format(_input1.shape))
print('The test data size after dropping Id feature is : {} '.format(_input0.shape))
sns.set_theme(rc={'grid.linewidth': 0.5, 'axes.linewidth': 0.75, 'axes.facecolor': '#fff3e9', 'axes.labelcolor': '#6b1000', 'figure.facecolor': '#f7e7da', 'xtick.labelcolor': '#6b1000', 'ytick.labelcolor': '#6b1000'})
train_missing = _input1.count().loc[_input1.count() < 1460].sort_values(ascending=False)
with plt.rc_context(rc={'figure.dpi': 120, 'axes.labelsize': 8.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    (fig, ax) = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(x=train_missing.values, y=train_missing.index, palette='viridis')
    plt.xlabel('Non-Na values')
test_missing = _input0.count().loc[_input0.count() < 1459].sort_values(ascending=False)
with plt.rc_context(rc={'figure.dpi': 120, 'axes.labelsize': 8.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    (fig, ax) = plt.subplots(1, 1, figsize=(7, 6))
    sns.barplot(x=test_missing.values, y=test_missing.index, palette='viridis')
    plt.xlabel('Non-Na values')
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
(fig, ax) = plt.subplots()
ax.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
sns.distplot(_input1['SalePrice'], fit=norm)