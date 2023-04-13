import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
pd.set_option('display.max_column', 200)
pd.set_option('display.max_rows', 1460)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input1.shape
_input1.dtypes.value_counts()
_input1.info()
_input1.describe()
missing_perc = _input1.isna().sum() / _input1.shape[0] * 100
with_miss = missing_perc[missing_perc > 0].sort_values(ascending=False)
with_miss
plt.figure(figsize=(12, 6))
plt.xticks(rotation=45)
sns.barplot(x=with_miss.index, y=with_miss)
plt.figure(figsize=(12, 6))
sns.distplot(_input1.SalePrice)
print('Skewness: %f' % _input1['SalePrice'].skew())

def plotHistProb():
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.distplot(_input1['SalePrice'], fit=norm)