import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.describe()
_input1.GrLivArea.describe()
for col in _input1.columns:
    print(col)
_input1.SalePrice.describe()
sns.displot(data=_input1, x='SalePrice', kind='ecdf', hue='YrSold', height=6, aspect=1.4, stat='count')
sns.countplot(x='MSZoning', data=_input1)
sns.countplot(x='HouseStyle', data=_input1)
_input1['YearBuilt'].describe()
sns.countplot(x='Foundation', data=_input1)
sns.catplot(x='BedroomAbvGr', kind='count', data=_input1, col='YrSold')
plt.figure(figsize=(20, 20))
sns.heatmap(_input1.corr(), annot=True, fmt='.1f', cmap='Blues')
_input1['BedroomAbvGr'] = pd.Categorical(_input1['BedroomAbvGr'])
plt.figure(figsize=(10, 10))
data2006 = _input1[_input1.YrSold == 2006]
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
_input1 = _input1.fillna(method='pad')
_input0 = _input0.fillna(method='pad')
print(_input1.isnull().sum().sum())
print(_input0.isnull().sum().sum())
train_null = find_nan_cols(_input1)
test_null = find_nan_cols(_input0)
print(train_null)
print(test_null)
_input1 = _input1.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], 1, inplace=False)
_input0 = _input0.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], 1, inplace=False)
print('NAN values in train_df:', _input1.isnull().sum().sum())
print('NAN values in test_df:', _input0.isnull().sum().sum())
y = _input1.SalePrice
X = _input1.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
x = _input0.select_dtypes(exclude=['object'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25)
model = DecisionTreeRegressor()