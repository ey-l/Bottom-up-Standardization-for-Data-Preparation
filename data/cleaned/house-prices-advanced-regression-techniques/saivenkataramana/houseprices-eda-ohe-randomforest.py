import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAERROR
from sklearn.preprocessing import OneHotEncoder
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
hp = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
hppred = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
hp.head()
hp.head()
print(hp.shape)
print(hppred.shape)
hp.isna().head()
hp.isnull().sum().sum()
hp.isnull().sum()
colnasum_tr = hp.isnull().sum().sort_values(ascending=False)
colnasum_pr = hppred.isnull().sum().sort_values(ascending=False)
print(colnasum_tr)
print(colnasum_pr)
hp.isnull().sum().plot()
hp.isnull().sum().iloc[0:20].plot()
hppred.isnull().sum().plot()
colna = colnasum_tr[colnasum_tr > 0]
print(colna)
colna.shape
hp_dna = hp.dropna(axis=1)
hppred_dna = hppred.dropna(axis=1)
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
hptrain_X.reset_index(inplace=True)
hptrain_X.pop('index')
hpval_X.reset_index(inplace=True)
hpval_X.pop('index')
hpval_X.head()
Xtr_int = hptrain_X.select_dtypes(include=int)
Xval_int = hpval_X.select_dtypes(include=int)
Xpr_int = hp_dna_prX.select_dtypes(include=int)
RFmodel = RandomForestRegressor(random_state=100)