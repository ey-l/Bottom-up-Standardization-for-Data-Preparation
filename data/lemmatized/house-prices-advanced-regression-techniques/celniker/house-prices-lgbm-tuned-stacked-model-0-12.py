import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tracemalloc
from pandas_summary import DataFrameSummary
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import lightgbm as lgb
from tqdm import tqdm
import octopus_ml as oc
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
warnings.simplefilter('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(4)
print('Train set: ', _input1.shape)
print('Test set: ', _input0.shape)
dfs = DataFrameSummary(_input1)
dfs.summary()
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 4))
df = pd.Series(1 - _input1.count() / len(_input1)).sort_values(ascending=False).head(20)
sns.barplot(x=df.index, y=df, palette='Blues_d')
plt.xticks(rotation=90)
_input1['YrSold'].value_counts()
grp_year = _input1.groupby('YrSold')
plt.figure(figsize=(5, 3))
df_years = grp_year['SalePrice'].mean().reset_index()
sns.barplot(x=df_years.YrSold, y=df_years['SalePrice'], palette='Blues_d')
plt.xticks(rotation=0)
from scipy.stats import norm, skew
sns.set_style('whitegrid')
plt.figure(figsize=(12, 4))
sns.distplot(_input1['SalePrice'], fit=norm)