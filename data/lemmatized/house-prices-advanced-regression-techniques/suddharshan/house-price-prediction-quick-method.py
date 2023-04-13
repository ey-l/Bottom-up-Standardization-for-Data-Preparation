import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
len(_input1.select_dtypes(include='object').columns)
_input1.head()
(_input1.shape, _input0.shape)
_input1.dtypes
_input1.info()
_input1.describe()
na = _input1.isna().sum() / len(_input1)
na[na > 0.5]
_input1 = _input1.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
_input0 = _input0.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
obj = _input1.select_dtypes(include='object').columns
plt.figure(figsize=(25, 20))
sns.heatmap(_input1.corr(), annot=True)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in obj:
    _input1[i] = _input1[i].fillna('aaaaaaaa')
    _input1[i] = _input1[i].astype(str)
for i in obj:
    _input0[i] = _input0[i].fillna('aaaaaaaa')
    _input0[i] = _input0[i].astype(str)
for i in obj:
    _input1[i] = label_encoder.fit_transform(_input1[i])
    _input0[i] = label_encoder.fit_transform(_input0[i])
obj
for i in _input1.select_dtypes(include='object').columns:
    print('____________________________________________________')
    print(i, _input1[i].nunique())
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0)
_input1 = pd.DataFrame(imp.fit_transform(_input1), columns=list(_input1.columns))
_input0 = pd.DataFrame(imp.fit_transform(_input0), columns=list(_input0.columns))
_input1
model = CatBoostRegressor(iterations=4000, verbose=False)
X = _input1.drop(columns=['SalePrice'])
y = _input1['SalePrice']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)