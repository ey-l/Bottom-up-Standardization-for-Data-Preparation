import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAERROR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print(_input1.shape)
print(_input0.shape)
_input1.isna().head()
_input1.isnull().sum().sum()
_input1.isnull().sum()
colnasum_tr = _input1.isnull().sum().sort_values(ascending=False)
colnasum_pr = _input0.isnull().sum().sort_values(ascending=False)
print(colnasum_tr)
print(colnasum_pr)
_input1.isnull().sum().plot()
_input1.isnull().sum().iloc[0:20].plot()
_input0.isnull().sum().plot()
colna = colnasum_tr[colnasum_tr > 0]
print(colna)
colna.shape
hp_dna = _input1.dropna(axis=1)
hppred_dna = _input0.dropna(axis=1)
hp_dna
hp_dna.isnull().sum().sum()
cols = hp_dna.columns
colsale = [col for col in cols if 'Sale' in col]
colsale
colspr = hppred_dna.columns
colsale = [col for col in colspr if 'Sale' in col]
colsale
pd.Series(cols).equals(colspr)
colsmatch = [col2 for col1 in cols for col2 in colspr if col1 == col2]
print(colsmatch)
print('')
print('NUmber of matching columns=', len(colsmatch))
hp_dna_X = hp_dna[colsmatch]
hp_dna_prX = hppred_dna[colsmatch]
hp_y = hp_dna['SalePrice']
(hptrain_X, hpval_X, hptrain_y, hpval_y) = train_test_split(hp_dna_X, hp_y, random_state=2)
hptrain_X = hptrain_X.reset_index(inplace=False)
hptrain_X.pop('index')
hpval_X = hpval_X.reset_index(inplace=False)
hpval_X.pop('index')
hpval_X.head()
Xtr_int = hptrain_X.select_dtypes(include=int)
Xval_int = hpval_X.select_dtypes(include=int)
Xpr_int = hp_dna_prX.select_dtypes(include=int)
RFmodel = RandomForestRegressor(random_state=100)