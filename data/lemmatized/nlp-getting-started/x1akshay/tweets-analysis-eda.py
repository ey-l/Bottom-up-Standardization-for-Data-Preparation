import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.head(6)
_input0.count()
_input0.tail(6)
_input0.dtypes
_input0.nunique()
_input0.shape
_input0.duplicated().sum()
_input0.isnull().sum()
_input0.isnull()
features_f = [f for f in _input0.columns if _input0[f].dtype == 'object']

def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (col, _input0[col].dtype, _input0[col].isnull().sum(), _input0[col].count(), _input0[col].nunique(), _input0[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows)
    df.columns = ['feature', 'dtype', 'nan', 'count', 'nunique', 'unique']
    return df
check(_input0[features_f])
plt.figure(figsize=(20, 10))
sns.heatmap(_input0[features_f].isnull(), cmap='Reds')
_input0['n_missing'] = _input0[features_f].isna().sum(axis=1)
_input0['n_missing'].value_counts().plot(kind='bar', title='Number of missing Values per data')
plt.figure(figsize=(20, 16))
corr = _input0.corr()
sns.heatmap(corr, cmap='YlGnBu', annot=True, square=True, vmax=0.8, linewidths=0.01, linecolor='white', annot_kws={'size': 12})
_input0.hist(figsize=(5, 5))