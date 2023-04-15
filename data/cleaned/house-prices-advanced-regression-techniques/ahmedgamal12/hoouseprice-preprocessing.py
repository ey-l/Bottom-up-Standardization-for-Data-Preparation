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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.shape
test.shape
submit.shape
train_null = train.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null.reset_index(inplace=True)
train_null.head(20)
test_null = test.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null.reset_index(inplace=True)
test_null.head(34)
train_null_array = np.array(train_null.iloc[:19]['index'])
train_null_array
test_null_array = np.array(test_null.iloc[:33]['index'])
test_null_array
train.columns
object_columns = train.select_dtypes(include='object').columns
object_columns
for i in object_columns:
    print(i)
    print(train[i].value_counts())
train.describe().T
train.describe(include='object').T
c = train.corr()
c
plt.figure(figsize=(20, 20))
sns.heatmap(c, annot=True)
correlation = c['SalePrice'].sort_values(ascending=False)
correlation = pd.DataFrame(correlation)
correlation.head(11)
train.drop(columns=['Id'], axis=1, inplace=True)
test.drop(columns=['Id'], axis=1, inplace=True)
train_null_array
test_null_array
arr_null = ['MiscFeature', 'Fence', 'PoolQC', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Alley']
arr_null
train.MasVnrType.value_counts()
combain = [train, test]
for i in arr_null:
    train[i].fillna('NA', inplace=True)
    test[i].fillna('NA', inplace=True)
train_null = train.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null.reset_index(inplace=True)
train_null.head(10)
test_null = test.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null.reset_index(inplace=True)
test_null.head(20)
train_null_array = list(set(train_null_array).symmetric_difference(arr_null))
train_null_array
test_null_array = list(set(test_null_array).symmetric_difference(arr_null))
test_null_array
for i in train_null_array:
    if train[i].dtype == 'object':
        train[i].fillna(train[i].mode()[0], inplace=True)
    else:
        train[i].fillna(train[i].median(), inplace=True)
for i in test_null_array:
    if test[i].dtype == 'object':
        test[i].fillna(test[i].mode()[0], inplace=True)
    else:
        test[i].fillna(test[i].median(), inplace=True)
train_null = train.isnull().sum()
train_null = pd.DataFrame(train_null, columns=['number']).sort_values(by='number', ascending=False)
train_null.reset_index(inplace=True)
train_null.head(5)
test_null = test.isnull().sum()
test_null = pd.DataFrame(test_null, columns=['number']).sort_values(by='number', ascending=False)
test_null.reset_index(inplace=True)
test_null.head(5)
train.drop(columns=['GarageCars'], axis=1, inplace=True)
test.drop(columns=['GarageCars'], axis=1, inplace=True)
(train.shape, test.shape)

def bar_chart(col):
    HousePrice = train.groupby([col])['SalePrice'].mean()
    df_HousePrice = pd.DataFrame(HousePrice).sort_values(by=['SalePrice'], ascending=False)
    df_HousePrice.reset_index(inplace=True)
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
plt.scatter(train['OverallQual'], train['SalePrice'])
plt.title('how OverallQual effect in SalePrice')
plt.xlabel('OverallQual')
plt.ylabel('HousePrice')
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')
plt.scatter(train['FullBath'], train['SalePrice'])
plt.title('how FullBath effect in SalePrice')
plt.xlabel('FullBath')
plt.ylabel('HousePrice')
plt.scatter(train['TotalBsmtSF'], train['SalePrice'])
plt.title('how TotalBsmtSF effect in SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('HousePrice')
plt.scatter(train['1stFlrSF'], train['SalePrice'])
plt.title('how 1stFlrSF effect in SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('HousePrice')
plt.scatter(train['TotRmsAbvGrd'], train['SalePrice'])
plt.title('how TotRmsAbvGrd effect in SalePrice')
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('HousePrice')
plt.scatter(train['LotFrontage'], train['SalePrice'])
plt.title('how LotFrontage effect in SalePrice')
plt.xlabel('LotFrontage')
plt.ylabel('HousePrice')
sns.distplot(x=train['SalePrice'])
sns.distplot(x=train['LotArea'])
sns.distplot(x=train['GrLivArea'])
sns.distplot(x=train['1stFlrSF'])
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for i in num_features:
    train[i] = np.log(train[i])
    test[i] = np.log(test[i])
for i in num_features:
    sns.distplot(train[i])


for i in num_features:
    sns.boxplot(train[i])


from sklearn.preprocessing import LabelEncoder
object_cols = train.select_dtypes(include='object')
le = LabelEncoder()
for i in object_cols:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)

def feature_selection(x, y, percent):
    x_top_columns = SelectPercentile(score_func=f_classif, percentile=percent)
    x_top_80 = x_top_columns.fit_transform(x, y)
    X_train_top_80 = list(x_train.columns[x_top_columns.get_support()])
    X_train_feature = x_train[x_train.columns[x_train.columns.isin(X_train_top_80)]]
    X_test_feature = x_test[x_test.columns[x_test.columns.isin(X_train_top_80)]]
    return (X_train_feature, X_test_feature)
(x_train_feature_80, x_test_feature_80) = feature_selection(x_train, y_train, 80)
feature_80_test = test[test.columns[test.columns.isin(x_train_feature_80)]]
feature_80_test
model_lr_feature_selection_80 = LinearRegression(normalize=True)