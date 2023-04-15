import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew
Train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
Test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Train.head()
Test.head()
(fig, ax) = plt.subplots()
ax.scatter(x=Train['GrLivArea'], y=Train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

Train = Train.drop(Train[(Train['GrLivArea'] > 4000) & (Train['SalePrice'] < 300000)].index)
(fig, ax) = plt.subplots()
ax.scatter(Train['GrLivArea'], Train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

sns.distplot(Train['SalePrice'], fit=norm)