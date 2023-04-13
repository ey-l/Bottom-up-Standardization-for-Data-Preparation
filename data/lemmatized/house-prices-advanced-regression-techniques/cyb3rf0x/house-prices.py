import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
all_data = pd.concat([_input1, _input0], axis=0)
print(_input0.shape, _input1.shape, all_data.shape)

def find_nan_cols(data):
    cols = []
    lenn = len(data)
    for col in data.columns:
        if data[col].count() < lenn:
            cols.append(col)
    return cols
find_nan_cols(all_data)
types_in_data = []
for i in _input1.columns:
    types_in_data.append(type(_input1[i][1386]))
set(types_in_data)

def data_types(train_d):
    t_dict = dict(ints=[], strs=[], floats=[])
    for i in _input1.columns:
        if i == 'Alley':
            t_dict['strs'].append(i)
        else:
            check = type(_input1[i][1386])
            if check == int or check == np.int64:
                t_dict['ints'].append(i)
            elif check == float or check == np.float64:
                t_dict['floats'].append(i)
            elif check == str:
                t_dict['strs'].append(i)
    return t_dict
types = data_types(_input1)

def mean_fillna(data, nan_cols, d_types):
    for col in nan_cols:
        if col in d_types['strs']:
            data[col] = data[col].fillna(value='None', inplace=False)
        elif col in d_types['ints']:
            mean = int(data[col].mean())
            data[col] = data[col].fillna(value=mean, inplace=False)
        else:
            mean = float(data[col].mean())
            data[col] = data[col].fillna(value=mean, inplace=False)
    return ('No NaNs more: ', len(find_nan_cols(data)) == 0)
mean_fillna(all_data, find_nan_cols(all_data), types)
all_data.head(10)

def categorical_data(dataset):
    for col in dataset.columns:
        if dataset.dtypes[col] == 'O':
            dataset[col] = dataset[col].astype('category')
            dataset[col] = dataset[col].cat.codes
categorical_data(all_data)
all_data.head(10)
Xx_train = all_data.iloc[:1460, 1:-1].values
Xx_test = all_data.iloc[1460:, 1:-1].values
y = all_data.iloc[:1460, -1].values
(x_train, x_test, y_train, y_test) = train_test_split(Xx_train, y, test_size=0.2, random_state=0)
model_rf = RandomForestRegressor(n_estimators=100, random_state=0)