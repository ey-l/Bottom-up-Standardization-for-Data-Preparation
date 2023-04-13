import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
df = pd.DataFrame()
df = pd.concat([_input1, _input0], axis=0)
df.shape
sns.heatmap(df.isna(), yticklabels=0)
df.info()
df = df.drop(columns=['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt'], axis=1)
df.shape
df.info()
df_object = df.select_dtypes(include=object)
df_num = df.select_dtypes(include=np.number)
df_object.info()
for i in df_object.columns:
    if df_object[i].isnull().sum() != 0:
        df_object[i] = df_object[i].fillna(df_object[i].mode()[0])
df_object.head()
df_object['MSZoning'].value_counts()
df_object.columns
df_object = pd.get_dummies(df_object, drop_first=True)
df_object.head()
df_object.shape
for i in df_num.columns:
    if df_num[i].isnull().sum() != 0:
        df_num[i] = df_num[i].fillna(df_num[i].mean())
d = pd.DataFrame()
d = pd.concat([df_object, df_num], axis=1)
d.head()
d.shape
sns.heatmap(d.isnull(), yticklabels=0)
d.head()
x_train = d.iloc[0:1460, :-1]
x_test = d.iloc[1460:, :-1]
y_train = d.iloc[0:1460, -1:]
(x_train.shape, x_test.shape, y_train.shape)
from xgboost import XGBRegressor
model = XGBRegressor()
from sklearn.feature_selection import SelectFromModel