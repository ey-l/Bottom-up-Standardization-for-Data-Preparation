import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'train df shape is {_input1.shape}')
print('-' * 50)
print(f'test df shape is {_input0.shape}')
_input1.head()
_input1.info()
missing_df = _input1.isnull().sum() / len(_input1)
missing_df[missing_df > 0.4]
missing_df_2 = _input0.isnull().sum() / len(_input0)
missing_df_2[missing_df_2 > 0.4]
_input1 = _input1.drop(columns=missing_df[missing_df > 0.4].index, axis=1, inplace=False)
_input0 = _input0.drop(columns=missing_df_2[missing_df_2 > 0.4].index, axis=1, inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.drop(columns='Id', axis=1, inplace=False)
numerical_features = [feature for feature in _input1.columns if _input1[feature].dtypes != 'O']
print(f'Number of Numerical Features are {len(numerical_features)}')
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtypes == 'O']
print(f'Number of Categorical Features are {len(categorical_features)}')
year_features = [feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]
year_features
for feature in year_features:
    print(feature, _input1[feature].unique())
for feature in year_features:
    data = _input1.copy()
    if feature != 'YrSold':
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature)
discrete_features = [feature for feature in numerical_features if len(_input1[feature].unique()) < 25 and feature not in year_features]
print('Discrete Variables Count: ', len(discrete_features))
continuous_features = [feature for feature in numerical_features if feature not in discrete_features and feature not in year_features]
print('Continuous Variables Count: ', len(continuous_features))
_input1[discrete_features].head()
for feature in discrete_features:
    data = _input1.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
for feature in continuous_features:
    data = _input1.copy()
    data[feature].hist()
    plt.xlabel(feature)
    plt.ylabel('Count')
for feature in continuous_features:
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data[feature].hist()
        plt.title(feature)
for feature in continuous_features:
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature)
for feature in continuous_features:
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
_input1[categorical_features].head()
for feature in categorical_features:
    data = _input1.copy()
    print(f'The feature is {feature} and no of categories are {len(data[feature].unique())}')
for feature in categorical_features:
    _input1.groupby(feature)['SalePrice'].median().plot(kind='bar')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
categorical_features_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 0 and _input1[feature].dtypes == 'O']
for feature in categorical_features_nan:
    print(f'{feature}: {np.round(_input1[feature].isnull().mean(), 4)}% missing values')

def replace_missing_nan_cat(dataset, features):
    data = dataset.copy()
    data[features] = data[features].fillna('Missing')
    return data
_input1 = replace_missing_nan_cat(_input1, categorical_features)
_input0 = replace_missing_nan_cat(_input0, categorical_features)
_input1[categorical_features].head(100)
_input0[categorical_features].head(100)
print(_input1[categorical_features_nan].isnull().sum())
numerical_features_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 0 and _input1[feature].dtypes != 'O']
numerical_features_nan
for feature in numerical_features_nan:
    _input1[feature] = _input1[feature].fillna(_input1[feature].median())
numerical_features_nan = [feature for feature in _input0.columns if _input0[feature].isnull().sum() > 0 and _input0[feature].dtypes != 'O']
numerical_features_nan
for feature in numerical_features_nan:
    _input0[feature] = _input0[feature].fillna(_input0[feature].median())
print(_input1[numerical_features_nan].isnull().sum())
print(_input0[numerical_features_nan].isnull().sum())
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    _input1[feature] = _input1['YrSold'] - _input1[feature]
    _input0[feature] = _input0['YrSold'] - _input0[feature]
_input1[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
_input0[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
num_continuous_features_log = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_continuous_features_log:
    _input1[feature] = np.log(_input1[feature])
num_continuous_features_log = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for feature in num_continuous_features_log:
    _input0[feature] = np.log(_input0[feature])
_input1[categorical_features].head(10)
for feature in categorical_features:
    temp = _input1.groupby(feature)['SalePrice'].count() / len(_input1)
    _input1[feature] = np.where(_input1[feature].isin(temp[temp > 0.1].index), _input1[feature], 'Others')
for feature in categorical_features:
    temp = _input0[feature].value_counts() / len(_input0)
    _input0[feature] = np.where(_input0[feature].isin(temp[temp > 0.1].index), _input0[feature], 'Others')
_input1['MSZoning'].unique()
_input0['MSZoning'].unique()
_input1.head(50)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
_input1[categorical_features] = enc.fit_transform(_input1[categorical_features])
_input0[categorical_features] = enc.fit_transform(_input0[categorical_features])
important_num_cols = list(_input1.corr()['SalePrice'][(_input1.corr()['SalePrice'] > 0.5) | (_input1.corr()['SalePrice'] < -0.5)].index)
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
important_cols = important_num_cols + cat_cols
_input1 = _input1[important_cols]
test_df_X = _input0[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']]
len(_input1.columns)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=85)
X_train.head()
test_df_X.head()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
models = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2 Score'])

def evaluation(y_test, predictions):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    return (mae, mse, r_squared)
lin_reg = LinearRegression()