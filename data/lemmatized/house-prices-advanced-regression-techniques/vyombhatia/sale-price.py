import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from category_encoders import CatBoostEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
y = _input1['SalePrice']
ID = _input0['Id']
_input1 = _input1.drop(['Id', 'SalePrice'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
nullcol = pd.DataFrame(list(_input1.isnull().sum()), columns=['examplesnull'])
cols = pd.DataFrame(_input1.columns, columns=['column name'])
nullinfo = pd.concat([cols, nullcol], axis=1)
s = _input1.dtypes == 'object'
catcol = list(s[s].index)
catsimp = SimpleImputer(strategy='most_frequent')
cattrain = pd.DataFrame(catsimp.fit_transform(_input1[catcol]), columns=catcol)
cattest = pd.DataFrame(catsimp.fit_transform(_input0[catcol]), columns=catcol)
n = _input1.dtypes == ('int64', 'float64')
numcol = list(n[n].index)
numsimp = SimpleImputer(strategy='median')
numtrain = pd.DataFrame(numsimp.fit_transform(_input1[numcol]), columns=numcol)
numtest = pd.DataFrame(numsimp.fit_transform(_input0[numcol]), columns=numcol)
enctrain = cattrain.copy()
LEnc = LabelEncoder()
for i in catcol:
    enctrain[i] = LEnc.fit_transform(enctrain[i])
LEnc = LabelEncoder()
enctest = cattest.copy()
for i in catcol:
    enctest[i] = LEnc.fit_transform(enctest[i])
newtrain = pd.concat([enctrain, numtrain], axis=1)
newtest = pd.concat([enctest, numtest], axis=1)
scale = MinMaxScaler()
scaledtrain = pd.DataFrame(newtrain, columns=newtrain.columns)
scaledtest = pd.DataFrame(newtest, columns=newtrain.columns)
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(scaledtrain, y, train_size=0.7, test_size=0.3)
RanModel = RandomForestRegressor(n_estimators=500)