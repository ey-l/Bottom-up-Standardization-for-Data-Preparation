import numpy as np
import pandas as pd
from dateutil.parser import parse
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.select_dtypes(include='object')

def process_df(df, cat_col_list=[]):
    new_df = _input1.copy()
    for col in _input1.select_dtypes(include='object').columns:
        if _input1[col].nunique() < 6 and col not in cat_col_list:
            new_df = pd.get_dummies(new_df, columns=[col])
        else:
            if col not in cat_col_list:
                cat_col_list.append(col)
            new_df[col] = _input1[col].astype('category').cat.codes
    return (new_df, cat_col_list)
(_input1, cat_col_list) = process_df(_input1)
pltdf = _input1.copy()
rename = [cname[0:10] for cname in _input1.columns]
pltdf.columns = rename
pltdf.iloc[:100, :24].plot(subplots=True, layout=(20, 4), figsize=(25, 20))
sns.displot(x=_input1['SalePrice'])
cols = []
cols_done = []
for col_one in _input1.iloc[:, :].columns:
    if _input1[col_one].corr(_input1['SalePrice']) > 0.6:
        cols.append(col_one)
    cols_done.append(col_one)
corrdf = _input1.copy()
corrdf = corrdf[cols].corr()
sns.heatmap(corrdf, cmap='Blues')
_input1[cols].iloc[:200, :].plot(subplots=True, layout=(7, 1), figsize=(25, 20))
g = sns.PairGrid(_input1[cols].iloc[:200, :], diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
tree_set = _input1.copy()
tree_set = tree_set.fillna(0)
target = tree_set['SalePrice']
tree_set = tree_set.drop(['SalePrice'], axis=1, inplace=False)
tree_clf = DecisionTreeRegressor(max_depth=3, random_state=1)