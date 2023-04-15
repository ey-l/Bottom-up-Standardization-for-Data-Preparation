import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas_summary as ps
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
from scipy.stats import norm
sns.set()
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
all_metrics = []
folder = '_data/input/house-prices-advanced-regression-techniques/'
train_df = pd.read_csv(folder + 'train.csv')
test_df = pd.read_csv(folder + 'test.csv')
sub_df = pd.read_csv(folder + 'sample_submission.csv')
print('train: ', train_df.shape)
print('test: ', test_df.shape)
print('sample_submission: ', sub_df.shape)
train_df.head()
train_df.info()
test_df.head()
test_df.info()
sub_df.head()
dfs = ps.DataFrameSummary(train_df)
print('categoricals: ', dfs.categoricals.tolist())
print('numerics: ', dfs.numerics.tolist())
dfs.summary()
dfs = ps.DataFrameSummary(test_df)
print('categoricals: ', dfs.categoricals.tolist())
print('numerics: ', dfs.numerics.tolist())
dfs.summary()
train_df.drop('Id', inplace=True, axis=1)
test_df.drop('Id', inplace=True, axis=1)
ps.DataFrameSummary(train_df[['SalePrice']]).summary().T
plt.figure(figsize=(12, 5))
sns.distplot(train_df['SalePrice'], fit=norm)