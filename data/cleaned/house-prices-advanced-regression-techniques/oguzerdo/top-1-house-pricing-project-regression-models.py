import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('fivethirtyeight')
from datetime import datetime
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p
from lightgbm import LGBMRegressor
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.simplefilter('ignore')
print('Setup complete')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train set size:', train.shape)
print('Test set size:', test.shape)
train.head()

def plot_numerical(col, discrete=False):
    if discrete:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')

def plot_categorical(col):
    (fig, ax) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[1])
    fig.suptitle(str(col) + ' analysis')
print('Plot functions are ready to use')
plt.figure(figsize=(8, 5))
a = sns.distplot(train.SalePrice, kde=False)
plt.title('SalePrice distribution')
a = plt.axvline(train.SalePrice.describe()['25%'], color='b')
a = plt.axvline(train.SalePrice.describe()['75%'], color='b')
print('SalePrice description:')
print(train.SalePrice.describe().to_string())
num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
num_features.remove('Id')
num_features.remove('SalePrice')
num_analysis = train[num_features].copy()
for col in num_features:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1, 1))
clf = ExtraTreesRegressor(random_state=42)