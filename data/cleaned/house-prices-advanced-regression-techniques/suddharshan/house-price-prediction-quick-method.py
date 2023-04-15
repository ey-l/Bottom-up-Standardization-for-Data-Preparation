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
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
len(df.select_dtypes(include='object').columns)
df.head()
(df.shape, df1.shape)
df.dtypes
df.info()
df.describe()
na = df.isna().sum() / len(df)
na[na > 0.5]
df = df.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
df1 = df1.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
obj = df.select_dtypes(include='object').columns
plt.figure(figsize=(25, 20))
sns.heatmap(df.corr(), annot=True)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in obj:
    df[i] = df[i].fillna('aaaaaaaa')
    df[i] = df[i].astype(str)
for i in obj:
    df1[i] = df1[i].fillna('aaaaaaaa')
    df1[i] = df1[i].astype(str)
for i in obj:
    df[i] = label_encoder.fit_transform(df[i])
    df1[i] = label_encoder.fit_transform(df1[i])
obj
for i in df.select_dtypes(include='object').columns:
    print('____________________________________________________')
    print(i, df[i].nunique())
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0)
df = pd.DataFrame(imp.fit_transform(df), columns=list(df.columns))
df1 = pd.DataFrame(imp.fit_transform(df1), columns=list(df1.columns))
df
model = CatBoostRegressor(iterations=4000, verbose=False)
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)