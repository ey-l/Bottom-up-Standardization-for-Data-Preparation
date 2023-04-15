import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.shape
df.columns
df = df.drop(['Id'], axis=1)
df.info()
df.describe(include='all')
col_with_NA_as_category = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
null = df.isnull().sum()
null = null[null.values > 0]
null.sort_values(ascending=False)
null_col = [i for i in null.index if i not in col_with_NA_as_category]
null_col
df[['LotFrontage']] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df[['LotFrontage']]))
df[['MasVnrArea']] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df[['MasVnrArea']]))
df[['MasVnrType']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df[['MasVnrType']]))
df[['Electrical']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df[['Electrical']]))
df[['GarageYrBlt']] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df[['GarageYrBlt']]))
for col in col_with_NA_as_category:
    df[col].fillna('Not Applicable', inplace=True)
df.isnull().sum()[df.isnull().sum().values > 0]
categorical_columns_with_numerical_values = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']
for i in categorical_columns_with_numerical_values:
    df[i] = df[i].astype(str)
corr = df.corr().sort_values(by='SalePrice', ascending=False)[['SalePrice']]
corr
col_to_be_removed = corr.index[7:]
df = df.drop(col_to_be_removed, axis=1)
numeric_col = [i for i in corr.index[1:] if i not in col_to_be_removed]
categorical_col = [i for i in df.columns[:-1] if i not in numeric_col]
print('No. of Categorical columns: ', len(categorical_col))
print('No. of Numerical columns: ', len(numeric_col))
k = 1
plt.figure(figsize=(10, 8))
for col in numeric_col:
    plt.subplot(2, 3, k)
    sns.boxplot(x=df[col])
    k += 1


def remove_outlier(column):
    p25 = column.describe()[4]
    p75 = column.describe()[6]
    IQR = p75 - p25
    ul = p75 + 1.5 * IQR
    ll = p25 - 1.5 * IQR
    column.mask(column > ul, ul, inplace=True)
    column.mask(column < ll, ll, inplace=True)
for col in numeric_col:
    if col != 'YearRemodAdd':
        remove_outlier(df[col])
k = 1
plt.figure(figsize=(10, 8))
for col in numeric_col:
    plt.subplot(2, 3, k)
    sns.boxplot(x=df[col])
    k += 1

df[numeric_col] = StandardScaler().fit_transform(df[numeric_col])

def bar_plot(col):
    d = df[['SalePrice']].groupby(df[col]).mean().reset_index()
    sns.barplot(x=d[col], y=d['SalePrice'], order=d.sort_values(by='SalePrice', ascending=True)[col], palette='Greens')
    plt.title(col + 'vs. SalePrice')
plt.figure(figsize=(20, 20))
i = 1
for col in categorical_columns_with_numerical_values:
    plt.subplot(4, 4, i)
    bar_plot(col)
    i += 1

plt.figure(figsize=(20, 45))
i = 1
for col in [j for j in categorical_col if j not in categorical_columns_with_numerical_values]:
    plt.subplot(9, 5, i)
    bar_plot(col)
    i += 1

drop = ['MoSold', 'BsmtHalfBath', 'BsmtFullBath', 'YrSold', 'LandSlope']
df = df.drop(drop, axis=1)
for i in drop:
    categorical_col.remove(i)
len(categorical_col)
for col in categorical_col:
    df[col] = LabelEncoder().fit_transform(df[col])
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=42, test_size=0.2)