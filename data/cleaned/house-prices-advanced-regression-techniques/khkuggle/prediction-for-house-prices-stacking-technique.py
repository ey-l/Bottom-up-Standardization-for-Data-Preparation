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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.shape
train.dtypes.value_counts()
train.info()
train.describe()
missing_perc = train.isna().sum() / train.shape[0] * 100
with_miss = missing_perc[missing_perc > 0].sort_values(ascending=False)
with_miss
plt.figure(figsize=(12, 6))
plt.xticks(rotation=45)
sns.barplot(x=with_miss.index, y=with_miss)
plt.figure(figsize=(12, 6))
sns.distplot(train.SalePrice)
print('Skewness: %f' % train['SalePrice'].skew())

def plotHistProb():
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.distplot(train['SalePrice'], fit=norm)