import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input1.shape
_input0.shape
_input2.shape
train_null = _input1.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null = train_null.reset_index(inplace=False)
train_null.head(20)
test_null = _input0.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null = test_null.reset_index(inplace=False)
test_null.head(34)
train_null_array = np.array(train_null.iloc[:19]['index'])
train_null_array
test_null_array = np.array(test_null.iloc[:33]['index'])
test_null_array
_input1.columns
object_columns = _input1.select_dtypes(include='object').columns
object_columns
for i in object_columns:
    print(i)
    print(_input1[i].value_counts())
_input1.describe().T
_input1.describe(include='object').T
c = _input1.corr()
c
plt.figure(figsize=(20, 20))
sns.heatmap(c, annot=True)
correlation = c['SalePrice'].sort_values(ascending=False)
correlation = pd.DataFrame(correlation)
correlation.head(11)
_input1 = _input1.drop(columns=['Id'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['Id'], axis=1, inplace=False)
train_null_array
test_null_array
arr_null = ['MiscFeature', 'Fence', 'PoolQC', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Alley']
arr_null
_input1.MasVnrType.value_counts()
combain = [_input1, _input0]
for i in arr_null:
    _input1[i] = _input1[i].fillna('NA', inplace=False)
    _input0[i] = _input0[i].fillna('NA', inplace=False)
train_null = _input1.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null = train_null.reset_index(inplace=False)
train_null.head(10)
test_null = _input0.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null = test_null.reset_index(inplace=False)
test_null.head(20)
train_null_array = list(set(train_null_array).symmetric_difference(arr_null))
train_null_array
test_null_array = list(set(test_null_array).symmetric_difference(arr_null))
test_null_array
for i in train_null_array:
    if _input1[i].dtype == 'object':
        _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
    else:
        _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
for i in test_null_array:
    if _input0[i].dtype == 'object':
        _input0[i] = _input0[i].fillna(_input0[i].mode()[0], inplace=False)
    else:
        _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
train_null = _input1.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null = train_null.reset_index(inplace=False)
train_null.head(5)
test_null = _input0.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null = test_null.reset_index(inplace=False)
test_null.head(5)
_input1 = _input1.drop(columns=['GarageCars'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['GarageCars'], axis=1, inplace=False)
(_input1.shape, _input0.shape)

def bar_chart(col):
    HousePrice = _input1.groupby([col])['SalePrice'].mean()
    df_HousePrice = pd.DataFrame(HousePrice).sort_values(by=['SalePrice'], ascending=False)
    df_HousePrice = df_HousePrice.reset_index(inplace=False)
    plt.bar(x=df_HousePrice[col], height=df_HousePrice['SalePrice'])
    plt.title(f'house price effect ber {col}')
    plt.xlabel(col)
    plt.ylabel('Price')
bar_chart('Fence')
bar_chart('GarageType')
bar_chart('RoofStyle')
plt.figure(figsize=(14, 6))
bar_chart('RoofMatl')
bar_chart('SaleType')
correlation.head(11)
plt.scatter(_input1['OverallQual'], _input1['SalePrice'])
plt.title('how OverallQual effect in SalePrice')
plt.xlabel('OverallQual')
plt.ylabel('HousePrice')
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')
plt.scatter(_input1['FullBath'], _input1['SalePrice'])
plt.title('how FullBath effect in SalePrice')
plt.xlabel('FullBath')
plt.ylabel('HousePrice')
plt.scatter(_input1['TotalBsmtSF'], _input1['SalePrice'])
plt.title('how TotalBsmtSF effect in SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('HousePrice')
plt.scatter(_input1['1stFlrSF'], _input1['SalePrice'])
plt.title('how 1stFlrSF effect in SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('HousePrice')
plt.scatter(_input1['TotRmsAbvGrd'], _input1['SalePrice'])
plt.title('how TotRmsAbvGrd effect in SalePrice')
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('HousePrice')
plt.scatter(_input1['LotFrontage'], _input1['SalePrice'])
plt.title('how LotFrontage effect in SalePrice')
plt.xlabel('LotFrontage')
plt.ylabel('HousePrice')
sns.distplot(x=_input1['SalePrice'])
sns.distplot(x=_input1['LotArea'])
sns.distplot(x=_input1['GrLivArea'])
sns.distplot(x=_input1['1stFlrSF'])
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for i in num_features:
    _input1[i] = np.log(_input1[i])
    _input0[i] = np.log(_input0[i])
for i in num_features:
    sns.distplot(_input1[i])
for i in num_features:
    sns.boxplot(_input1[i])
from sklearn.preprocessing import LabelEncoder
object_cols = _input1.select_dtypes(include='object')
le = LabelEncoder()
for i in object_cols:
    _input1[i] = le.fit_transform(_input1[i])
    _input0[i] = le.fit_transform(_input0[i])
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
X = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)

def feature_selection(x, y, percent):
    x_top_columns = SelectPercentile(score_func=f_classif, percentile=percent)
    x_top_80 = x_top_columns.fit_transform(x, y)
    X_train_top_80 = list(x_train.columns[x_top_columns.get_support()])
    X_train_feature = x_train[x_train.columns[x_train.columns.isin(X_train_top_80)]]
    X_test_feature = x_test[x_test.columns[x_test.columns.isin(X_train_top_80)]]
    return (X_train_feature, X_test_feature)
(x_train_feature_80, x_test_feature_80) = feature_selection(x_train, y_train, 80)
feature_80_test = _input0[_input0.columns[_input0.columns.isin(x_train_feature_80)]]
feature_80_test
model_lr_feature_selection_80 = LinearRegression(normalize=True)