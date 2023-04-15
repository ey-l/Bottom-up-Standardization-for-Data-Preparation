import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train.head()
train.shape
train.isnull().sum()
sns.heatmap(train.isnull(), yticklabels=False, cmap='plasma')
train.isnull().sum().sort_values(ascending=False)[0:19]
test.isnull().sum().sort_values(ascending=False)[0:33]
train['SalePrice'].describe()
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
sns.distplot(train['SalePrice'], fit=norm)