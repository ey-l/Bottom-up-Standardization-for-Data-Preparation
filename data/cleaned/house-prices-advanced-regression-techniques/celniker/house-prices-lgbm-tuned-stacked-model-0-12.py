
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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(4)
print('Train set: ', train_df.shape)
print('Test set: ', test_df.shape)
dfs = DataFrameSummary(train_df)
dfs.summary()
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 4))
df = pd.Series(1 - train_df.count() / len(train_df)).sort_values(ascending=False).head(20)
sns.barplot(x=df.index, y=df, palette='Blues_d')
plt.xticks(rotation=90)
train_df['YrSold'].value_counts()
grp_year = train_df.groupby('YrSold')
plt.figure(figsize=(5, 3))
df_years = grp_year['SalePrice'].mean().reset_index()
sns.barplot(x=df_years.YrSold, y=df_years['SalePrice'], palette='Blues_d')
plt.xticks(rotation=0)
from scipy.stats import norm, skew
sns.set_style('whitegrid')
plt.figure(figsize=(12, 4))
sns.distplot(train_df['SalePrice'], fit=norm)