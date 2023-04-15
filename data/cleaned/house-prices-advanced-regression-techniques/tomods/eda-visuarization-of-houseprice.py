import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns
sns.set_style('white')
import time
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print('<train>')
print('Number of rows:', train.shape[0], 'Number of colums:', train.shape[1])

print('<test>')
print('Number of rows:', test.shape[0], 'Number of colums:', test.shape[1])

train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)
train['flag'] = 'train'
test['flag'] = 'test'
alldata = pd.concat([train, test], axis=0).reset_index(drop=True)
alldata.shape
df = alldata
plt.figure(figsize=(30, 16))
plt.title('Missing Value')
sns.heatmap(df.isnull(), cbar=False)
df = alldata
total = df.isnull().sum()
percent = round(df.isnull().sum() / df.isnull().count() * 100, 2)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Ratio_of_NA(%)'])
types = pd.DataFrame(df[missing_data.index].dtypes, columns=['Types'])
missing_data = pd.concat([missing_data, types], axis=1)
missing_data = missing_data.sort_values('Total', ascending=False)
categorical_cols = missing_data[missing_data['Types'] == 'object'].index
numerical_cols = missing_data[missing_data['Types'] != 'object'].index
print('Top 20 columns for Missing value')
print(missing_data.head(20))
print()
print('---Types---')
print(set(missing_data['Types']))
print()
print('---Categorical col---')
print(categorical_cols)
print()
print('---Numerical col---')
print(numerical_cols)
print()
print('---Categorical col with NaN---')
print(missing_data[(missing_data['Types'] == 'object') & (missing_data['Ratio_of_NA(%)'] > 0)].index)
print()
print('---Numerical col with NaN---')
print(missing_data[(missing_data['Types'] != 'object') & (missing_data['Ratio_of_NA(%)'] > 0)].index)
from scipy.stats import norm, skew
plt.figure(figsize=(12, 6))
sns.distplot(train['SalePrice'], fit=norm)