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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
pd.pandas.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(_input1.shape)
_input1.head()
_input1.info()
missing_count = _input1.isnull().sum()
features_with_missing_values = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1]
for feature in features_with_missing_values:
    print(feature, ' has ', missing_count[feature], ' missing values')
data = _input1.copy()
for feature in features_with_missing_values:
    data[feature] = np.where(data[feature].isnull(), 'missing value', 'legit value')
    data.groupby(feature)['SalePrice'].mean().plot.bar()
    plt.title(feature)
for feature in _input1.columns:
    print(feature, ' -- ', _input1[feature].dtype)
numerical_features = [feature for feature in _input1.columns if _input1[feature].dtype in ['int64', 'float64']]
print('the number of numerical columns == ', len(numerical_features))
year_numerical_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_numerical_feature
for feature in year_numerical_feature:
    print(feature, len(_input1[feature].unique()))
    print(feature, _input1[feature].unique())
    print()
import seaborn as sns
data = _input1.copy()
for feature in year_numerical_feature:
    if feature not in ['YrSold']:
        data[feature] = data['YrSold'] - data[feature]
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(data[feature], data['SalePrice'])
        plt.subplot(1, 2, 2)
        sns.histplot(data=data, x=feature, y='SalePrice', kde=False)
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
discrete_numerical_features = [feature for feature in numerical_features if len(_input1[feature].unique()) < 25 and feature not in year_numerical_feature + ['Id']]
discrete_numerical_features
data = _input1.copy()
for feature in discrete_numerical_features:
    data.groupby(feature)['SalePrice'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
continuous_numerical_feature = [feature for feature in numerical_features if feature not in discrete_numerical_features + year_numerical_feature + ['Id']]
len(continuous_numerical_feature)
data = _input1.copy()
for feature in continuous_numerical_feature:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
for feature in continuous_numerical_feature:
    _input1.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
data = _input1.copy()
for feature in continuous_numerical_feature:
    if 0 in data[feature].unique():
        continue
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtypes in ['object']]
categorical_features
for feature in categorical_features:
    print('The feature {} has {} many categories'.format(feature, len(_input1[feature].unique())))
data = _input1.copy()
for feature in categorical_features:
    data.groupby(feature)['SalePrice'].mean().plot.bar(rot=0)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
_input1.head()
X = _input1.drop(['Id', 'SalePrice'], axis=1)
y = _input1.SalePrice
from sklearn.model_selection import train_test_split
(x_train, x_val, y_train, y_val) = train_test_split(X, y, random_state=1, test_size=0.1)
print(x_train.shape)
print(x_val.shape)
cat_features_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() >= 1 and _input1[feature].dtypes in ['object']]
x_train[cat_features_with_nan].isnull().sum()

def replace_cat_features(dataset, features_with_nan):
    data = _input1.copy()
    data[features_with_nan] = data[features_with_nan].fillna('Missing')
    return data
x_train = replace_cat_features(x_train, cat_features_with_nan)
x_val = replace_cat_features(x_val, cat_features_with_nan)
x_train[cat_features_with_nan].isnull().sum()
x_val[cat_features_with_nan].isnull().sum()
numerical_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() >= 1 and _input1[feature].dtypes in ['int64', 'float64']]
len(numerical_with_nan)
train_num_nan = x_train[numerical_with_nan].isnull().sum()
for col in train_num_nan.index:
    print('{}    {}'.format(col, train_num_nan[col]))
val_num_nan = x_val[numerical_with_nan].isnull().sum()
for col in val_num_nan.index:
    print('{}    {}'.format(col, val_num_nan[col]))
for feature in numerical_with_nan:
    if train_num_nan[feature] > 0:
        x_train[feature] = x_train[feature].fillna(x_train[feature].median(), inplace=False)
    if val_num_nan[feature] > 0:
        x_val[feature] = x_val[feature].fillna(x_val[feature].median(), inplace=False)
x_train[numerical_with_nan].isnull().sum()
x_val[numerical_with_nan].isnull().sum()
year_numerical_feature
year_numerical_feature.remove('YrSold')
for feature in year_numerical_feature:
    x_train[feature] = x_train['YrSold'] - x_train[feature]
    x_val[feature] = x_val['YrSold'] - x_val[feature]
x_train[year_numerical_feature].head()
continuous_numerical_feature
x_train.head()
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
train_encoded = pd.DataFrame(cat_encoder.fit_transform(x_train[categorical_features]))
val_encoded = pd.DataFrame(cat_encoder.transform(x_val[categorical_features]))
train_encoded.index = x_train.index
val_encoded.index = x_val.index
num_x_train = x_train.drop(categorical_features, axis=1)
num_x_val = x_val.drop(categorical_features, axis=1)
encoded_x_train = pd.concat([num_x_train, train_encoded], axis=1)
encoded_x_val = pd.concat([num_x_val, val_encoded], axis=1)
encoded_x_train.head()
encoded_x_val.head()
encoded_x_train.shape
encoded_x_val.shape
scaler_cols = encoded_x_train.columns
scaler = MinMaxScaler()
x_train_encoded_scaled = pd.DataFrame(scaler.fit_transform(encoded_x_train), columns=scaler_cols)
x_val_encoded_scaled = pd.DataFrame(scaler.transform(encoded_x_val), columns=scaler_cols)
x_train_encoded_scaled.head()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel_x_train = SelectFromModel(Lasso(alpha=0.01, random_state=1, max_iter=10000000))