import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import sklearn
import sklearn.feature_selection
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
Y = _input1['SalePrice']
_input1 = _input1.drop(['SalePrice'], axis=1)
(N, M) = _input1.shape
print('Number of Samples', N, 'Number of Features', M)
_input1.head()
(N_2, m_2) = _input0.shape
_input0.head()
percent_missing = _input1.isnull().mean() * 100
percent_missing = percent_missing.sort_values()
plt.figure(figsize=(15, 6))
plt.bar(np.arange(len(percent_missing)), percent_missing)
plt.ylabel('Percentage of Missing Values')
plt.xticks(np.arange(len(percent_missing)), percent_missing.keys(), rotation=90)
mutual = []
train_x = np.zeros((N, 1))
test_x = np.zeros((N_2, 1))
for name in _input1.columns:
    if _input1[name].dtypes == 'O':
        le = preprocessing.LabelEncoder()
        _input1[name] = _input1[name].fillna('missing', inplace=False)
        _input1[name] = _input1[name].astype(str)
        Canadate = le.fit_transform(_input1[name])
        Canadate = Canadate.reshape(-1, 1)
        m = sklearn.feature_selection.mutual_info_regression(Canadate, Y)[0]
        mutual += [m]
        if m > 0.1:
            print('Variable Added')
            print('Type:', _input1[name].dtypes, 'Variable', name, 'Most Frequent Label:', _input1[name].mode().values, 'Number of Labels:', len(_input1[name].unique()))
            One_hot = OneHotEncoder()
            train_x = np.append(train_x, One_hot.fit_transform(Canadate).toarray(), axis=1)
            _input0[name] = _input0[name].fillna(_input1[name].mode().values[0], inplace=False)
            _input0[name] = _input0[name].astype(str)
            Canadate = le.transform(_input0[name])
            Canadate = Canadate.reshape(-1, 1)
            test_x = np.append(test_x, One_hot.transform(Canadate).toarray(), axis=1)
    elif _input1[name].dtypes == 'int' or _input1[name].dtypes == 'float':
        norm = preprocessing.RobustScaler()
        _input1[name] = _input1[name].fillna(_input1[name].mean(), inplace=False)
        Canadate = norm.fit_transform(_input1[name].values.reshape(-1, 1))
        Canadate[Canadate > 4] = 4
        Canadate[Canadate < -2] = -2
        m = sklearn.feature_selection.mutual_info_regression(Canadate, Y)[0]
        mutual += [m]
        if m > 0.1:
            print('Variable Added')
            print('Type:', _input1[name].dtypes, 'Variable', name, 'Average Value', _input1[name].mean())
            train_x = np.append(train_x, Canadate, axis=1)
            _input0[name] = _input0[name].fillna(_input1[name].mean(), inplace=False)
            Canadate = norm.transform(_input0[name].values.reshape(-1, 1))
            Canadate[Canadate > 4] = 4
            Canadate[Canadate < -2] = -2
            test_x = np.append(test_x, Canadate, axis=1)
train_x = train_x[:, 1:]
test_x = test_x[:, 1:]
print('Final Training Shape:', train_x.shape)
print('Final Test Shape:', test_x.shape)
plt.figure(figsize=(15, 6))
plt.bar(np.arange(79), mutual)
plt.xticks(np.arange(79), _input1.columns, rotation=90)
plt.ylabel('Mutual Information')
Y = np.log1p(Y)
(X_train, X_val, y_train, y_val) = train_test_split(train_x, Y, test_size=0.1, random_state=42)
import xgboost as xgb
lr = linear_model.ElasticNetCV(cv=5, n_alphas=10, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1])
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20000)