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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.columns
_input1.head()
_input1.shape
_input1.isnull().sum()
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='plasma')
_input1.isnull().sum().sort_values(ascending=False)[0:19]
_input0.isnull().sum().sort_values(ascending=False)[0:33]
_input1['SalePrice'].describe()
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
var = 'GrLivArea'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'TotalBsmtSF'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'OverallQual'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
sns.distplot(_input1['SalePrice'], fit=norm)