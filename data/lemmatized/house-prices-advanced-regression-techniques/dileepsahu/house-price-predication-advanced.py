import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
_input1.head(5)
feature_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1]
for feature in feature_with_nan:
    print(feature, np.round(_input1[feature].isnull().mean(), 4), '% missing values')
for feature in feature_with_nan:
    df = _input1.copy()
    df[feature] = np.where(df[feature].isnull(), 1, 0)
    plt.figure(figsize=(10, 5))
    df.groupby(feature)['SalePrice'].mean().plot.bar()
    plt.ylabel('Sales price')
    plt.legend(['1 for missing values'])
print('Id of the house {}'.format(len(_input1.Id)))
numerical_features = [feature for feature in _input1.columns if df[feature].dtypes != 'O']
print('Number of numerical variables {}'.format(len(numerical_features)))
_input1[numerical_features].head()
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
print(year_feature)
for feature in year_feature:
    print(feature, _input1[feature].unique())
_input1.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price VS YearSold')
for feature in year_feature:
    if feature != 'YrSold':
        df = _input1.copy()
        df[feature] = df['YrSold'] - df[feature]
        plt.scatter(df[feature], df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
discrete_feature = [feature for feature in numerical_features if len(df[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print('Discrete vaiables count: {}'.format(len(discrete_feature)))
discrete_feature
_input1[discrete_feature].head()
for feature in discrete_feature:
    df = _input1.copy()
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
continous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
print('Number of continous feature {}'.format(len(continous_feature)))
for feature in continous_feature:
    df = _input1.copy()
    df[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
for feature in continous_feature:
    df = _input1.copy()
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df['SalePrice'] = np.log(df['SalePrice'])
        plt.scatter(df[feature], df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
for feature in continous_feature:
    df = _input1.copy()
    if 0 in _input1[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtypes == 'O']
categorical_features
_input1[categorical_features].head()
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature, len(df[feature].unique())))
for feature in categorical_features:
    df = _input1.copy()
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalesPrice')
    plt.title(feature)
    plt.title(feature)
feature_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes == 'O']

def replace_cat_feature(df, feature_nan):
    data = df.copy()
    data[feature_nan] = data[feature_nan].fillna('Missing')
    return data
_input1 = replace_cat_feature(_input1, feature_nan)
_input1[feature_nan].isnull().sum()
_input1.head()
numerical_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes != 'O']
for feature in numerical_with_nan:
    print('{}: {}% missing values'.format(feature, np.around(_input1[feature].isnull().mean(), 4)))
for feature in numerical_with_nan:
    median_value = _input1[feature].median()
    _input1[feature + 'nan'] = np.where(_input1[feature].isnull(), 1, 0)
    _input1[feature] = _input1[feature].fillna(median_value, inplace=False)
_input1[numerical_with_nan].isnull().sum()
_input1.head()
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    _input1[feature] = _input1['YrSold'] - _input1[feature]
_input1.head()
_input1[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]
_input1.head()
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    _input1[feature] = np.log(df[feature])
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=False)
_input1.head()
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtypes == 'O']
categorical_features
for feature in categorical_features:
    temp = _input1.groupby(feature)['SalePrice'].count() / len(_input1)
    temp_df = temp[temp > 0.01].index
    _input1[feature] = np.where(_input1[feature].isin(temp_df), _input1[feature], 'Rare_var')
_input1.head()
for feature in categorical_features:
    labels_ordered = _input1.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    _input1[feature] = _input1[feature].map(labels_ordered)
feature_scale = [feature for feature in _input1.columns if feature not in ['id', 'SalePrice']]
scaler = MinMaxScaler()