import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
import warnings
warnings.simplefilter('ignore')
print('Setup complete')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()

def plot_numerical(col, discrete=False):
    if discrete:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.stripplot(x=col, y='SalePrice', data=_input1, ax=ax[0])
        sns.countplot(_input1[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=col, y='SalePrice', data=_input1, ax=ax[0])
        sns.distplot(_input1[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')

def plot_categorical(col):
    (fig, ax) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.stripplot(x=col, y='SalePrice', data=_input1, ax=ax[0])
    sns.boxplot(x=col, y='SalePrice', data=_input1, ax=ax[1])
    fig.suptitle(str(col) + ' analysis')
print('plot_numerical() & plot_categorical() are ready to use')
plt.figure(figsize=(8, 5))
a = sns.distplot(_input1.SalePrice, kde=False)
plt.title('SalePrice distribution')
a = plt.axvline(_input1.SalePrice.describe()['25%'], color='b')
a = plt.axvline(_input1.SalePrice.describe()['75%'], color='b')
print('SalePrice description:')
print(_input1.SalePrice.describe().to_string())
num_features = [col for col in _input1.columns if _input1[col].dtype in ['int64', 'float64']]
num_features.remove('Id')
num_features.remove('SalePrice')
num_analysis = _input1[num_features].copy()
for col in num_features:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1, 1))
clf = ExtraTreesRegressor(random_state=42)