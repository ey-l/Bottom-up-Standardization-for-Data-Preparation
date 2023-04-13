import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
pd.set_option('display.max_columns', None)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
from sklearn.feature_selection import mutual_info_regression
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
_input1.head()
_input1.isnull().sum()
from sklearn.impute import SimpleImputer
df_most_frequent = _input1.copy()
mean_imputer = SimpleImputer(strategy='most_frequent')
df_most_frequent.iloc[:, :] = mean_imputer.fit_transform(df_most_frequent)
df_most_frequent.isnull().sum()
from sklearn.preprocessing import LabelEncoder
for c in _input1.columns:
    if _input1[c].dtype == 'float16' or _input1[c].dtype == 'float32' or _input1[c].dtype == 'float64':
        _input1[c].fillna(_input1[c].mean())
_input1 = _input1.fillna(-999)
for f in _input1.columns:
    if _input1[f].dtype == 'object':
        lbl = LabelEncoder()