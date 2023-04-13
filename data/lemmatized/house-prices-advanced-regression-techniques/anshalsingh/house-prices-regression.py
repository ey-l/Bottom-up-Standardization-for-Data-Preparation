import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.isnull().sum().sum()
col = _input1.columns
null_col = []
for i in col:
    if _input1[i].isnull().any() == True:
        print(i, ':', _input1[i].isnull().sum())
        null_col.append(i)
col_test = _input0.columns
null_col_test = []
for i in col_test:
    if _input0[i].isnull().any() == True:
        print(i, ':', _input0[i].isnull().sum())
        null_col_test.append(i)
target = _input1['SalePrice']
target[:10]
missing_object = []
for i in _input1.select_dtypes(include='object').columns:
    if i in null_col:
        missing_object.append(i)
missing_object
_input1.shape
_input1 = _input1.drop(columns=['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'Id'])
_input0 = _input0.drop(columns=['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'Id'])
_input1.head()
cols = _input1.columns
print(cols)
cols_test = _input0.columns
_input1 = _input1.drop(columns=['SalePrice'])
_input1.head()
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='most_frequent')
temp_train = impute.fit_transform(_input1)
temp_test = impute.fit_transform(_input0)
temp_train = pd.DataFrame(temp_train)
temp_train.insert(75, column='75', value=target)
temp_train = pd.DataFrame(temp_train)
temp_train.columns = cols
temp_test = pd.DataFrame(temp_test)
temp_test.columns = cols_test
temp_train.isnull().sum().sum()
temp_test.isnull().sum().sum()
temp_train.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in temp_train.select_dtypes(include='object').columns:
    temp_train[i] = le.fit_transform(temp_train[i])
for i in temp_test.select_dtypes(include='object').columns:
    temp_test[i] = le.fit_transform(temp_test[i])
print(temp_train.shape, temp_test.shape)
temp_train.describe()
temp_train.head()
un_prices = temp_train.SalePrice.sort_values(ascending=False)
un_prices[:10]
sns.histplot(target, log_scale=True, color='green', kde=True)
un_prices[:10].plot(kind='barh', color='orange')
sns.relplot(data=temp_train, y='SaleCondition', x=target, hue='SaleCondition')
temp_train.head()
sns.pairplot(temp_train[['Street', 'LotShape', 'LotArea', 'LotFrontage', 'PoolArea', 'SaleType', 'SalePrice']], diag_kind='hist')
data = temp_train.drop(columns=['SalePrice'])
a1 = data.columns
a2 = temp_test.columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(data)
test_df = sc.fit_transform(temp_test)
data = pd.DataFrame(data)
data.columns = a1
data.head()
test_df = pd.DataFrame(test_df)
test_df.columns = a2
test_df.head()
from sklearn.model_selection import train_test_split
(x_tr, x_te, y_tr, y_te) = train_test_split(data, target, random_state=42, test_size=0.2)
print(x_tr.shape, y_tr.shape)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
grid_params = {'weights': ['uniform', 'distance'], 'n_neighbors': [5, 10, 15, 20]}
gs = GridSearchCV(KNeighborsRegressor(), grid_params, verbose=1, cv=3)