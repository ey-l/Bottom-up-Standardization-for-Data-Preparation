import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_data.head(6)
train_data.count()
train_data.tail(6)
train_data.dtypes
train_data.nunique()
train_data.shape
train_data.duplicated().sum()
train_data.isnull().sum()
train_data.isnull()
features_f = [f for f in train_data.columns if train_data[f].dtype == 'object']

def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (col, train_data[col].dtype, train_data[col].isnull().sum(), train_data[col].count(), train_data[col].nunique(), train_data[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows)
    df.columns = ['feature', 'dtype', 'nan', 'count', 'nunique', 'unique']
    return df
check(train_data[features_f])
plt.figure(figsize=(20, 10))
sns.heatmap(train_data[features_f].isnull(), cmap='Reds')

train_data['n_missing'] = train_data[features_f].isna().sum(axis=1)
train_data['n_missing'].value_counts().plot(kind='bar', title='Number of missing Values per data')

plt.figure(figsize=(20, 16))
corr = train_data.corr()
sns.heatmap(corr, cmap='YlGnBu', annot=True, square=True, vmax=0.8, linewidths=0.01, linecolor='white', annot_kws={'size': 12})

train_data.hist(figsize=(5, 5))