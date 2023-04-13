import numpy as np
import pandas as pd
import random
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.dtypes
_input1.isnull().sum()
_input0.isnull().sum()
train_df = _input1.select_dtypes(include=[np.number]).interpolate()
train_df.head()
train_df.isnull().sum()
train_df.shape
test_df = _input0.select_dtypes(include=[np.number]).interpolate()
test_df.head()
test_df.isnull().sum()
test_df.shape
col = ['SalePrice', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond']
sns.set(style='ticks')
sns.pairplot(train_df[col], size=3, kind='reg')
import matplotlib.pyplot as plt
corrmat = _input1.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, vmax=0.8, square=True)
import seaborn as sns
sns.distplot(train_df['SalePrice'])
y = np.log(train_df.SalePrice)
X = train_df.drop(['SalePrice', 'Id'], axis=1)
(X.shape, y.shape)
from sklearn.model_selection import train_test_split
(X, X_test, y, y_test) = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()