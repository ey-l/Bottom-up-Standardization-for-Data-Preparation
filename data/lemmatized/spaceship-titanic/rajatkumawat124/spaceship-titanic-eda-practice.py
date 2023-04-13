import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1.describe()
_input1.isna().sum()
sns.heatmap(_input1.isna(), cbar=False)
_input1.shape
_input1['Floor'] = _input1['Cabin'].str[:1]
_input1['CabinPort'] = _input1['Cabin'].str[-1:]
all_cols = _input1.columns
for col in all_cols:
    print(f'{col} has {_input1[col].isna().sum()} null values and {_input1[col].nunique()} unique values')
_input1 = _input1.drop(['Name', 'Cabin'], axis=1)
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Floor', 'CabinPort']
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
_input1 = _input1.dropna(subset=cat_cols, inplace=False)
(fig, axes) = plt.subplots(2, 3, figsize=(25, 10))
for (i, col) in zip(range(6), num_cols):
    sns.stripplot(ax=axes[i // 3][i % 3], x='CabinPort', y=col, data=_input1, palette='GnBu', hue='Transported')
    axes[i // 3][i % 3].set_title(f'{col} Stripplot')
for col in num_cols:
    p_mean = _input1[col].loc[_input1['CabinPort'] == 'P'].mean()
    s_mean = _input1[col].loc[_input1['CabinPort'] == 'S'].mean()
    _input1.loc[(_input1['CabinPort'] == 'P') & _input1[col].isna(), col] = p_mean
    _input1.loc[(_input1['CabinPort'] == 'S') & _input1[col].isna(), col] = s_mean
for col in num_cols:
    print(f'{col} has {_input1[col].isna().sum()} null values')
(fig, axes) = plt.subplots(2, 3, figsize=(25, 10))
for (i, col) in zip(range(6), num_cols):
    sns.boxplot(ax=axes[i // 3][i % 3], x='Transported', y=col, data=_input1, palette='GnBu', hue='Transported')
    axes[i // 3][i % 3].set_title(f'{col} Boxplot')

def outlier_limits(df, col_name, q1=0.25, q3=0.75):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)

def no_of_outliers(df, variable, q1=0.25, q3=0.75):
    (low_limit, up_limit) = outlier_limits(df, variable, q1=q1, q3=q3)
    return df.loc[(df[variable] < low_limit) | (df[variable] > up_limit), variable].count()
for col in num_cols:
    count = no_of_outliers(_input1, col)
    print(f'{col} has {count} outliers')

def replace_with_limits(df, variable, q1=0.25, q3=0.75):
    (low_limit, up_limit) = outlier_limits(df, variable, q1=q1, q3=q3)
    df.loc[df[variable] < low_limit, variable] = low_limit
    df.loc[df[variable] > up_limit, variable] = up_limit
for col in num_cols:
    replace_with_limits(_input1, col)
    count = no_of_outliers(_input1, col)
    print(f'{col} has {count} outliers')
(fig, axes) = plt.subplots(2, 3, figsize=(25, 10))
for (i, col) in zip(range(6), num_cols):
    sns.histplot(ax=axes[i // 3][i % 3], x=col, data=_input1, palette='GnBu', hue='Transported', bins=5, multiple='dodge')
    axes[i // 3][i % 3].set_title(f'{col} Countplot')
encoded_train = _input1.copy()
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Floor']:
    encoded_col = encoded_train.groupby(col).size() / len(encoded_train)
    encoded_train[col] = encoded_train[col].apply(lambda x: encoded_col[x])
encoded_train.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_vip = le.fit_transform(encoded_train['VIP'])
encoded_transported = le.fit_transform(encoded_train['Transported'])
encoded_train['VIP'] = encoded_vip
encoded_train['Transported'] = encoded_transported
encoded_train.head()
corr = encoded_train[1:].corr(method='spearman')
plt.figure(figsize=(20, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.title('Spearman Correlation Heatmap')
figure = plt.figure(figsize=(20, 30))
sns.pairplot(data=_input1, vars=num_cols, hue='Transported', palette='GnBu')
(fig, axes) = plt.subplots(1, 3, figsize=(20, 7))
for (i, col) in zip(range(3), num_cols[0:3]):
    sns.scatterplot(ax=axes[i], x='Age', y=col, hue='Transported', style='VIP', data=encoded_train, palette='GnBu')
(fig, axes) = plt.subplots(1, 2, figsize=(20, 7))
for (i, col) in zip(range(2), num_cols[0:5]):
    sns.scatterplot(ax=axes[i], x='Age', y=col, hue='Transported', style='VIP', data=encoded_train, palette='GnBu')
(fig, axes) = plt.subplots(2, 3, figsize=(25, 10))
for (i, col) in zip(range(6), num_cols):
    sns.stripplot(ax=axes[i // 3][i % 3], x='Transported', y=col, data=_input1, palette='GnBu', hue='VIP', jitter=True)
    axes[i // 3][i % 3].set_title(f'{col} Stripplot')
sns.histplot(x='Transported', data=_input1, palette='GnBu', hue='VIP', bins=2, multiple='dodge')
plt.title(f'VIPs Countplot')