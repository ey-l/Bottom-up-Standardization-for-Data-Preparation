import numpy as np
import pandas as pd
from scipy.stats import norm, skew

import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
housing = data_train.copy()
corr_matrix = housing.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
(f, ax) = plt.subplots(figsize=(11, 9))
plt.title('Correlation matrix of numerical features')
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
print("Correlation coefficient 'GarageYrBlt' / 'YearBuilt': '{:.1%}".format(corr_matrix.loc['GarageYrBlt', 'YearBuilt']))
print("Correlation coefficient '1stFlrSF' / 'TotalBsmtSF': '{:.1%}".format(corr_matrix.loc['1stFlrSF', 'TotalBsmtSF']))
print("Correlation coefficient 'TotRmsAbvGrd' / 'GrLivArea': '{:.1%}".format(corr_matrix.loc['TotRmsAbvGrd', 'GrLivArea']))
print("Correlation coefficient 'GarageArea' / 'GarageCars': '{:.1%}".format(corr_matrix.loc['GarageArea', 'GarageCars']))
corr_matrix = corr_matrix.drop('GarageYrBlt')
corr_matrix = corr_matrix.drop('1stFlrSF')
corr_matrix = corr_matrix.drop('TotRmsAbvGrd')
corr_matrix = corr_matrix.drop('GarageArea')
plt.figure(figsize=(50, 6))
plt.xticks(rotation=45)
sns.barplot(x=corr_matrix['SalePrice'].sort_values(ascending=False).index, y=corr_matrix['SalePrice'].sort_values(ascending=False))
corr_matrix = corr_matrix.drop('SalePrice')
relevant_num_features = corr_matrix['SalePrice'].sort_values(ascending=False).loc[:'BsmtUnfSF'].index
relevant_num_features
sns.distplot(housing['SalePrice'])
print('Skewness: %f' % housing['SalePrice'].skew())
print('Kurtosis: %f' % housing['SalePrice'].kurt())
housing['SalePrice'] = np.log1p(housing['SalePrice'])
sns.distplot(housing['SalePrice'], fit=norm)
n_nan = []
for col in housing.columns:
    n_nan.append(int(housing[col].isna().sum(axis=0)) / len(housing.index))
nan_percentage = pd.DataFrame(n_nan, index=housing.columns)
nan_percentage = nan_percentage.iloc[:, 0].sort_values(ascending=False)
nan_percentage[:10]
low_relevant_cat = nan_percentage[:4].index
low_relevant_cat
data_train['TotalSF'] = data_train['TotalBsmtSF'] + data_train['1stFlrSF'] + data_train['2ndFlrSF']
data_test['TotalSF'] = data_test['TotalBsmtSF'] + data_test['1stFlrSF'] + data_test['2ndFlrSF']
y_train = np.log1p(data_train['SalePrice'])
data_train = data_train.drop('SalePrice', axis=1)

def select_relevant_data(df, relevant_num_features, low_relevant_cat):
    df_cat = df.select_dtypes(include=object)
    df_num = df.select_dtypes(include=np.number)
    df_cat = df_cat.drop(low_relevant_cat, axis=1)
    df_num = df_num.loc[:, relevant_num_features]
    return pd.concat([df_cat, df_num], axis=1)
X_train = select_relevant_data(data_train, relevant_num_features, low_relevant_cat)
X_test = select_relevant_data(data_test, relevant_num_features, low_relevant_cat)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='-1')), ('OneHotEncoder', OneHotEncoder())])
X_train_cat = X_train.select_dtypes(include=object)
X_train_num = X_train.select_dtypes(include=np.number)
num_attribs = list(X_train_num)
cat_attribs = list(X_train_cat)
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)])
X_fit = X_train.append(X_test, ignore_index=True)