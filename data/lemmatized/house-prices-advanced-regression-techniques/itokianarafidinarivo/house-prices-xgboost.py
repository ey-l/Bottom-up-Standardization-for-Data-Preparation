import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)

def basic_insights(frame, label=None):
    space = 20
    if label is not None:
        print(f"{'Name'.rjust(space)} : {label}")
    print(f"{'Dimensions'.rjust(space)} : {frame.shape} <=> Total = {frame.shape[0] * frame.shape[1]}")
    print(f"{'Missing values'.rjust(space)} : {frame.isna().sum().sum()} <=> {frame.isna().sum().sum() / (frame.shape[0] * frame.shape[1]) * 100}%")
basic_insights(_input1, label='Train')
basic_insights(_input0, label='Test')
_input1.isna().sum()[_input1.isna().sum() > 0] / _input1.shape[0] * 100
cols = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
_input1 = _input1.drop(columns=cols)
_input0 = _input0.drop(columns=cols)
convert_1 = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
convert_2 = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
convert_3 = {'Y': 1, 'N': 0}
map_ordinal = {'ExterQual': convert_1, 'ExterCond': convert_1, 'BsmtQual': convert_1, 'BsmtCond': convert_1, 'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, 'BsmtFinType1': convert_2, 'BsmtFinType2': convert_2, 'HeatingQC': convert_1, 'CentralAir': convert_3, 'KitchenQual': convert_1, 'FireplaceQu': convert_1, 'GarageFinish': {'Unf': 0, 'RFn': 1, 'Fin': 2}, 'GarageQual': convert_1, 'GarageCond': convert_1, 'PoolQC': convert_1, 'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}}
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype
all_data = pd.concat([_input1.copy(), _input0.copy()], axis=0)
all_data = all_data.replace(map_ordinal)
categories = dict()
for cat in all_data.dtypes[all_data.dtypes == 'O'].index:
    categories[cat] = all_data[cat].dropna().unique().tolist()
del all_data

def preprocessing(frame, scaler, is_train=False, target=None, submission=False):
    if is_train:
        frame[target] = frame[target].apply(np.log)
    frame = frame.replace(map_ordinal)
    numericals = frame.dtypes[frame.dtypes != 'O'].index.tolist()
    categoricals = frame.dtypes[frame.dtypes == 'O'].index.tolist()
    if is_train:
        numericals.remove(target)
    frame[numericals] = frame[numericals].fillna(frame[numericals].median())
    frame[numericals] = pd.DataFrame(scaler.fit_transform(frame[numericals]), columns=numericals) if is_train else pd.DataFrame(scaler.transform(frame[numericals]), columns=numericals)
    for key in categories.keys():
        categorical_type = CategoricalDtype(categories=categories[key], ordered=False)
        frame[key] = frame[key].astype(categorical_type)
    frame = pd.concat([frame.drop(columns=categoricals), pd.get_dummies(frame[categoricals])], axis=1)
    if is_train:
        X = frame.drop(columns=target)
        y = frame[target]
        if submission:
            return (X, y)
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.33)
        return (X_train, X_test, y_train, y_test)
    else:
        X = frame
        return X
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
(X_train, X_test, y_train, y_test) = preprocessing(_input1.copy(), scaler, is_train=True, target='SalePrice')
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'eta': 0.3}
model = XGBRegressor(**params)