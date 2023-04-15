import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import skew, kurtosis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from scipy.special import boxcox1p
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
description = open('_data/input/house-prices-advanced-regression-techniques/data_description.txt').read()
test_id = df_test['Id']
df_test.head()
df_train.info()
df_train.dtypes.value_counts()
desc_dict = {}
for radek in description.splitlines():
    if radek.split(':')[0] in df_train.columns:
        desc_dict[radek.split(':')[0]] = radek.split(':')[1]

def des_object(df, varname):
    table = vcounts = df[varname].value_counts()
    vcounts_len = len(vcounts)
    std = round(float(df[[varname, 'SalePrice']].groupby(varname).agg(['mean']).std()), 0)
    description = desc_dict[str(varname)] if varname in desc_dict.keys() else ''
    print('')
    print('*********************************')
    plt.figure(figsize=(16, 5))
    sns.set_style('whitegrid')
    plt.subplot(121)
    plt.axis([0, 10, 0, 10])
    i = 7.5
    plt.text(0, 8.5, f'there is: {vcounts_len} different values', horizontalalignment='left', fontsize=12)
    for row in range(np.min([len(table), 10])):
        text = table.index[row] + ': ' + str(table.iloc[row])
        plt.text(0, i, text, horizontalalignment='left', fontsize=12)
        i -= 0.5
    plt.text(0, i - 0.5, f'std: {std}', horizontalalignment='left', fontsize=12)
    plt.text(0, 9.5, description, horizontalalignment='left', fontsize=12)
    plt.text(0, 9, '-------------------------------------------------------', horizontalalignment='left', fontsize=12)
    plt.title(varname + '| type:' + str(df[varname].dtype), loc='left', weight='bold')
    plt.axis('off')
    plt.subplot(122)
    g = sns.boxenplot(x=varname, y='SalePrice', data=df, showfliers=False)
    sns.pointplot(x=varname, y='SalePrice', data=df, linestyles='--', scale=0.4, color='k', capsize=0)
    g.set_title(varname)
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()

    print('*********************************')
    print('')

def des_numeric(df, varname):
    print('*********************************')
    table = pd.DataFrame(df[varname].describe().round(2))
    skw = skew(df[varname], axis=0, bias=True)
    kts = kurtosis(df[varname], axis=0, bias=True)
    description = desc_dict[str(varname)] if varname in desc_dict.keys() else ''
    sns.set_style('whitegrid')
    plt.figure(figsize=(16, 5))
    plt.subplot(131)
    plt.axis([0, 10, 0, 10])
    i = 8.5
    for row in range(len(table)):
        text = table.index[row] + ': ' + str(table.iloc[row, 0])
        plt.text(0, i, text, horizontalalignment='left', fontsize=12)
        i -= 0.5
    plt.text(0, 9.5, description, horizontalalignment='left', fontsize=12)
    plt.text(0, 9, '-------------------------------------------------------', horizontalalignment='left', fontsize=12)
    plt.text(0, i, f'NA values: {df[varname].isna().sum()}', horizontalalignment='left', fontsize=12)
    plt.text(0, i - 0.5, f'unique values: {df[varname].nunique()}', horizontalalignment='left', fontsize=12)
    plt.text(0, i - 1.5, f'skew: {round(skw, 2)}', horizontalalignment='left', fontsize=12)
    plt.text(0, i - 2, f'kurtosis: {round(kts, 2)}', horizontalalignment='left', fontsize=12)
    plt.title(varname + '| type:' + str(df[varname].dtype), loc='left', weight='bold')
    plt.axis('off')
    plt.subplot(132)
    g = sns.histplot(data=df[varname], alpha=1, kde=True)
    g.set_title('Histogram')
    plt.subplot(133)
    g1 = sns.boxplot(data=df[varname], palette=['#7FFF00'])
    g1.set_title('Boxplot')
    plt.xticks([])
    plt.tight_layout()


def des_df(df):
    for c in df.columns:
        if df[c].dtype == object:
            des_object(df, c)
        else:
            des_numeric(df, c)
des_df(df_train.select_dtypes(exclude=['object']))
df_cat_only_train = df_train.select_dtypes(include=['object']).copy()
df_cat_only_train.loc[:, 'SalePrice'] = df_train['SalePrice'].copy()
des_df(df_cat_only_train)
df_concat = pd.concat([df_train, df_test])
missing_vals = pd.Series(df_concat.isna().sum()).reset_index().rename(columns={'index': 'Feature', 0: 'Missing Values'})
sns.set(rc={'figure.figsize': (10, 12)})
g = sns.barplot(data=missing_vals[missing_vals['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False), x='Missing Values', y='Feature')

def handle_na_values(df):
    for col in list(df.select_dtypes(exclude=['object']).columns):
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    df = df.fillna({k: f'no_{k}' for k in df})
    return df
df_train = handle_na_values(df_train)
df_test = handle_na_values(df_test)
df_concat = handle_na_values(df_concat)
df_concat['GarageYrBlt'].describe()
df_concat.loc[df_concat['GarageYrBlt'] > 2010, 'GarageYrBlt'] = 2007
df_concat[df_concat['YrSold'] < df_concat['YearRemodAdd']][['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]
df_concat['YearRemodAdd'] = np.where(df_concat['YrSold'] < df_concat['YearRemodAdd'], df_concat['YrSold'], df_concat['YearRemodAdd'])
df_concat['YrSold'] = np.where(df_concat['YrSold'] < df_concat['YearBuilt'], df_concat['YearBuilt'], df_concat['YrSold'])
df_concat['age_ToS'] = df_concat['YrSold'] - df_concat['YearBuilt']
df_concat['ageRm_ToS'] = df_concat['YrSold'] - df_concat['YearRemodAdd']
df_concat['age_garage_sold'] = df_concat['YrSold'] - df_concat['GarageYrBlt']
df_concat = df_concat.drop(['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1)
ordinal_cat = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'Fence']
for variable in ordinal_cat:
    for value in set(df_test.select_dtypes(include=['object'])[variable]):
        if not value in set(df_train.select_dtypes(include=['object'])[variable]):
            print(variable, value)
df_concat.loc[df_concat['KitchenQual'] == 'no_KitchenQual', 'KitchenQual'] = 'Fa'
for col in ordinal_cat:
    oe = OrdinalEncoder(categories=[list(df_train.groupby([col])['SalePrice'].mean().sort_values().index)])
    df_concat.loc[:, col] = oe.fit_transform(df_concat[[col]])
oe = OrdinalEncoder(categories=[['no_GarageCond', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
df_concat.loc[:, 'GarageCond'] = oe.fit_transform(df_concat[['GarageCond']])
oe = OrdinalEncoder(categories=[['no_PoolQC', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
df_concat.loc[:, 'PoolQC'] = oe.fit_transform(df_concat[['PoolQC']])
cols_to_dum = df_concat.select_dtypes(include=['object']).columns
df_dummies = pd.get_dummies(df_concat[cols_to_dum], drop_first=True)
df_concat = df_concat.drop(cols_to_dum, axis=1)
df_final = pd.concat([df_concat, df_dummies], axis=1)
df_final = df_final.drop('Id', axis=1).copy()
df_final['SalePrice'] = np.log1p(df_final['SalePrice'])
df_final.loc[:, ~df_final.columns.isin(['SalePrice'])] = boxcox1p(df_final.loc[:, ~df_final.columns.isin(['SalePrice'])], 0.15)

def Split_Df(df, train_rows):
    train = df.iloc[:train_rows].copy()
    test = df.iloc[train_rows:].copy().drop('SalePrice', axis=1)
    return (train, test)
(df_train, df_test) = Split_Df(df_final, len(df_train))
corrmat = df_train.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
plt.figure(figsize=(14, 10))
g = sns.heatmap(df_train[top_corr_features].corr(), annot=True, cmap='RdYlGn')
y = df_train['SalePrice']
X = df_train.loc[:, ~df_train.columns.isin(['SalePrice'])]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LassoCV
lasso = Lasso(alpha=0.0005)