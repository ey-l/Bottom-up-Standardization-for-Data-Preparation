import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
_input1.describe()
_input1.isnull().sum()
_input1.info()
_input1 = _input1.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'])
cols = _input1.columns[_input1.isnull().any()].tolist()
cols
df1 = pd.DataFrame()
df2 = pd.DataFrame()
l = []
r = []
for i in _input1:
    if np.dtype(_input1[i]) == 'object':
        l.append(i)
        df1[i] = _input1[i]
    else:
        df2[i] = _input1[i]
        if np.dtype(_input1[i]) == 'int64':
            r.append(i)
df2.info()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)