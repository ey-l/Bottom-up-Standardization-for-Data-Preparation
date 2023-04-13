import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
matplotlib.style.use('ggplot')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.describe()
_input0.describe()
corr = _input1.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1, square=True)
k = 20
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(12, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
from scipy.stats import norm, skew
from scipy import stats
sns.distplot(_input1['SalePrice'], fit=norm)