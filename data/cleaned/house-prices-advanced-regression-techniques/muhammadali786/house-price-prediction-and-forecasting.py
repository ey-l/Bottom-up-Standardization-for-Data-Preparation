import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.describe()
df_train.GrLivArea.describe()
for col in df_train.columns:
    print(col)
df_train.SalePrice.describe()
sns.displot(data=df_train, x='SalePrice', kind='ecdf', hue='YrSold', height=6, aspect=1.4, stat='count')
sns.countplot(x='MSZoning', data=df_train)
sns.countplot(x='HouseStyle', data=df_train)
df_train['YearBuilt'].describe()
sns.countplot(x='Foundation', data=df_train)
sns.catplot(x='BedroomAbvGr', kind='count', data=df_train, col='YrSold')
plt.figure(figsize=(20, 20))
sns.heatmap(df_train.corr(), annot=True, fmt='.1f', cmap='Blues')
df_train['BedroomAbvGr'] = pd.Categorical(df_train['BedroomAbvGr'])
plt.figure(figsize=(10, 10))
data2006 = df_train[df_train.YrSold == 2006]
plt.scatter(x=data2006['YearBuilt'], y=data2006['GrLivArea'], s=data2006['SalePrice'] / 500, c=data2006['BedroomAbvGr'].cat.codes, cmap='Accent', alpha=0.6, edgecolors='white', linewidth=2)
plt.xlabel('Year Built')
plt.ylabel('House Size')
plt.title('Year 2006')
plt.ylim(0, 6000)
plt.xlim(1870, 2010)
plt.legend()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def find_nan_cols(df):
    null = df.isnull().sum()
    missing_df = pd.concat([null], axis=1, keys=['nancount'])
    return missing_df[missing_df.nancount > 0]
df_train = df_train.fillna(method='pad')
df_test = df_test.fillna(method='pad')
print(df_train.isnull().sum().sum())
print(df_test.isnull().sum().sum())
train_null = find_nan_cols(df_train)
test_null = find_nan_cols(df_test)
print(train_null)
print(test_null)
df_train.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], 1, inplace=True)
df_test.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], 1, inplace=True)
print('NAN values in train_df:', df_train.isnull().sum().sum())
print('NAN values in test_df:', df_test.isnull().sum().sum())
y = df_train.SalePrice
X = df_train.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
x = df_test.select_dtypes(exclude=['object'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)
model = DecisionTreeRegressor()