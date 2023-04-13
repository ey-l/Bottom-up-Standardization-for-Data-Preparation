import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import math
import warnings as wr
wr.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
print('duplicated rows =', _input1.duplicated().sum())
print('columns containing missing values =', _input1.isnull().any().sum())
sale_price = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1)
total_data = pd.concat((_input1, _input0), axis=0)
total_data.head()
print('duplicated rows =', total_data.duplicated().sum())
print('columns containing missing values =', total_data.isnull().any().sum())
total_data.info()
data_object = total_data.select_dtypes('object')
print('object shape ', data_object.shape)
data_num = total_data.select_dtypes(['int64', 'float64'])
print('num shape ', data_num.shape)
data_num.describe()
data_object.describe()
for i in data_object.columns:
    print(data_object[i].value_counts())
for i in data_num.columns:
    print(data_num[i].value_counts())
total_data['ExterQual'] = total_data['ExterQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
total_data['ExterCond'] = total_data['ExterCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
total_data['BsmtQual'] = total_data['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
total_data['BsmtCond'] = total_data['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
total_data['BsmtExposure'] = total_data['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
total_data['HeatingQC'] = total_data['HeatingQC'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
total_data['KitchenQual'] = total_data['KitchenQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
total_data['FireplaceQu'] = total_data['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
total_data['GarageFinish'] = total_data['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
total_data['GarageQual'] = total_data['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
total_data['GarageCond'] = total_data['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
total_data['PavedDrive'] = total_data['PavedDrive'].replace({'Y': 3, 'P': 2, 'N': 1})
total_data['PoolQC'] = total_data['PoolQC'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0})
total_data['Fence'] = total_data['Fence'].replace({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
total_data
print('columns containing missing values =', total_data.isnull().any().sum())
missing_counts = pd.DataFrame(total_data.isnull().sum().sort_values(ascending=False))
missing_columns = missing_counts[missing_counts.iloc[:, 0] > 0]
missing_columns
plt.figure(figsize=(20, 10))
missing_columns = missing_counts[missing_counts.iloc[:, 0] > 0]
sns.barplot(x=missing_columns.index, y=missing_columns.iloc[:, 0])
plt.xticks(rotation=90)
total_data.isnull().sum().sort_values(ascending=False)
drop_col = list(missing_counts[missing_counts.iloc[:, 0] > 1000].index)
try:
    total_data = total_data.drop(columns=drop_col, axis=0)
    missing_columns = missing_columns.drop(index=drop_col, axis=1)
except:
    pass
print(total_data[missing_columns.index].info())
missing_object = total_data[missing_columns.index].select_dtypes('object').columns
print('missing object', len(missing_object))
missing_num = total_data[missing_columns.index].select_dtypes(['int64', 'float64']).columns
print('missing num ', len(missing_num))
for i in missing_num:
    total_data[i] = total_data[i].fillna(total_data[i].median())
for j in missing_object:
    total_data[j] = total_data[j].fillna(total_data[j].mode()[0])
print(total_data.isnull().any().sum())
data_all_processed_x = pd.get_dummies(total_data)
data_all_processed_x
training_data = data_all_processed_x.head(1460)
testing_data = data_all_processed_x.tail(1459)
training_data
training_data
sale_price
(x_train, x_test, y_train, y_test) = train_test_split(training_data, sale_price, test_size=0.2, random_state=40)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn import tree
model = tree.DecisionTreeRegressor()