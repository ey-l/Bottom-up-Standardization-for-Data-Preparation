import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_train.head()
data_train.shape
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data_test.head()
data_train.describe()
data_train.isnull().sum()
data_train.info()
data_train = data_train.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'])
cols = data_train.columns[data_train.isnull().any()].tolist()
cols
df1 = pd.DataFrame()
df2 = pd.DataFrame()
l = []
r = []
for i in data_train:
    if np.dtype(data_train[i]) == 'object':
        l.append(i)
        df1[i] = data_train[i]
    else:
        df2[i] = data_train[i]
        if np.dtype(data_train[i]) == 'int64':
            r.append(i)
df2.info()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)