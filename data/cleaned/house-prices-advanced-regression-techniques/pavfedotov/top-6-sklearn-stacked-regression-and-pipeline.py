from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, LassoLars, Lasso, RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
(fig, ax) = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000) | (train['SalePrice'] < 36000)].index)
(fig, ax) = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)


def plot_dist(var):
    sns.distplot(var, fit=norm)