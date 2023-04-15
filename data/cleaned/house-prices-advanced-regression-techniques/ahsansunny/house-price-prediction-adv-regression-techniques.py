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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'train df shape is {train_df.shape}')
print('-' * 50)
print(f'test df shape is {test_df.shape}')
train_df.head()
train_df.info()
missing_df = train_df.isnull().sum() / len(train_df)
missing_df[missing_df > 0.4]
missing_df_2 = test_df.isnull().sum() / len(test_df)
missing_df_2[missing_df_2 > 0.4]
train_df.drop(columns=missing_df[missing_df > 0.4].index, axis=1, inplace=True)
test_df.drop(columns=missing_df_2[missing_df_2 > 0.4].index, axis=1, inplace=True)
train_df.isnull().sum()
test_df.isnull().sum()
train_df.drop(columns='Id', axis=1, inplace=True)
numerical_features = [feature for feature in train_df.columns if train_df[feature].dtypes != 'O']
print(f'Number of Numerical Features are {len(numerical_features)}')
categorical_features = [feature for feature in train_df.columns if train_df[feature].dtypes == 'O']
print(f'Number of Categorical Features are {len(categorical_features)}')
year_features = [feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]
year_features
for feature in year_features:
    print(feature, train_df[feature].unique())
for feature in year_features:
    data = train_df.copy()
    if feature != 'YrSold':
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature)

discrete_features = [feature for feature in numerical_features if len(train_df[feature].unique()) < 25 and feature not in year_features]
print('Discrete Variables Count: ', len(discrete_features))
continuous_features = [feature for feature in numerical_features if feature not in discrete_features and feature not in year_features]
print('Continuous Variables Count: ', len(continuous_features))
train_df[discrete_features].head()
for feature in discrete_features:
    data = train_df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')

for feature in continuous_features:
    data = train_df.copy()
    data[feature].hist()
    plt.xlabel(feature)
    plt.ylabel('Count')

for feature in continuous_features:
    data = train_df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data[feature].hist()
        plt.title(feature)

for feature in continuous_features:
    data = train_df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature)

for feature in continuous_features:
    data = train_df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)

train_df[categorical_features].head()
for feature in categorical_features:
    data = train_df.copy()
    print(f'The feature is {feature} and no of categories are {len(data[feature].unique())}')
for feature in categorical_features:
    train_df.groupby(feature)['SalePrice'].median().plot(kind='bar')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')

categorical_features_nan = [feature for feature in train_df.columns if train_df[feature].isnull().sum() > 0 and train_df[feature].dtypes == 'O']
for feature in categorical_features_nan:
    print(f'{feature}: {np.round(train_df[feature].isnull().mean(), 4)}% missing values')

def replace_missing_nan_cat(dataset, features):
    data = dataset.copy()
    data[features] = data[features].fillna('Missing')
    return data
train_df = replace_missing_nan_cat(train_df, categorical_features)
test_df = replace_missing_nan_cat(test_df, categorical_features)
train_df[categorical_features].head(100)
test_df[categorical_features].head(100)
print(train_df[categorical_features_nan].isnull().sum())
numerical_features_nan = [feature for feature in train_df.columns if train_df[feature].isnull().sum() > 0 and train_df[feature].dtypes != 'O']
numerical_features_nan
for feature in numerical_features_nan:
    train_df[feature] = train_df[feature].fillna(train_df[feature].median())
numerical_features_nan = [feature for feature in test_df.columns if test_df[feature].isnull().sum() > 0 and test_df[feature].dtypes != 'O']
numerical_features_nan
for feature in numerical_features_nan:
    test_df[feature] = test_df[feature].fillna(test_df[feature].median())
print(train_df[numerical_features_nan].isnull().sum())
print(test_df[numerical_features_nan].isnull().sum())
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    train_df[feature] = train_df['YrSold'] - train_df[feature]
    test_df[feature] = test_df['YrSold'] - test_df[feature]
train_df[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
test_df[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
num_continuous_features_log = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_continuous_features_log:
    train_df[feature] = np.log(train_df[feature])
num_continuous_features_log = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for feature in num_continuous_features_log:
    test_df[feature] = np.log(test_df[feature])
train_df[categorical_features].head(10)
for feature in categorical_features:
    temp = train_df.groupby(feature)['SalePrice'].count() / len(train_df)
    train_df[feature] = np.where(train_df[feature].isin(temp[temp > 0.1].index), train_df[feature], 'Others')
for feature in categorical_features:
    temp = test_df[feature].value_counts() / len(test_df)
    test_df[feature] = np.where(test_df[feature].isin(temp[temp > 0.1].index), test_df[feature], 'Others')
train_df['MSZoning'].unique()
test_df['MSZoning'].unique()
train_df.head(50)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
train_df[categorical_features] = enc.fit_transform(train_df[categorical_features])
test_df[categorical_features] = enc.fit_transform(test_df[categorical_features])
important_num_cols = list(train_df.corr()['SalePrice'][(train_df.corr()['SalePrice'] > 0.5) | (train_df.corr()['SalePrice'] < -0.5)].index)
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
important_cols = important_num_cols + cat_cols
train_df = train_df[important_cols]
test_df_X = test_df[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']]
len(train_df.columns)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
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