import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import pandas_profiling
import seaborn as sns
import matplotlib.style as style
import matplotlib.pyplot as plt

from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
from scipy.stats import norm, skew
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')
house_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Dataset shape:', 'house_train', house_train.shape, 'house_test', house_test.shape)
plt.figure(figsize=(8, 12))
house_train.corr()['SalePrice'].sort_values().plot(kind='barh')
k = 10
cols = house_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
k_corr_matrix = house_train[cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(k_corr_matrix, annot=True, cmap=plt.cm.RdBu_r)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(house_train[cols], size=2)

target = house_train['SalePrice']
(f, axes) = plt.subplots(1, 3, figsize=(15, 4))
sns.distplot(target, kde=False, fit=stats.johnsonsu, ax=axes[0])
sns.distplot(target, kde=False, fit=stats.norm, ax=axes[1])
sns.distplot(target, kde=False, fit=stats.lognorm, ax=axes[2])
house_train['SalePrice'] = np.log1p(house_train['SalePrice'])