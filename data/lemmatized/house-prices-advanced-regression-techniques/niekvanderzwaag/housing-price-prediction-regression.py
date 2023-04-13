import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [12, 6]
sns.set(style='darkgrid')
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
sns.scatterplot(data=_input1, x='GrLivArea', y='SalePrice')
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 200000)].index)
sns.scatterplot(data=_input1, x='GrLivArea', y='SalePrice')
sns.distplot(_input1['SalePrice'], fit=norm)