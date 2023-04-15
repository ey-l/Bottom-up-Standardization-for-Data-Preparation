import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def MAPE(y_true, y_pred):
    (y_true, y_pred) = (np.array(y_true), np.array(y_pred))
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
testing = test.copy()
target = 'SalePrice'
df = pd.concat([train, test], axis=0)
df.info()
df.head(5)
for column in df.columns:
    try:
        df[column] = pd.to_numeric(df[column])
    except ValueError:
        pass
df.drop(columns=['Id'], inplace=True)
plt.subplots(figsize=(10, 5))
sns.distplot(df[target].dropna(), kde=True, label=target, color='green', bins=100)
plt.legend(prop={'size': 12})

plt.figure(figsize=(12, 10))
(df.loc[:, df.isnull().any()].isna().sum() / df.shape[0]).sort_values().plot(kind='barh', label='% of missing values')
plt.axvline(x=0.2, color='r', linestyle='--', label='Reference line')
plt.legend()
plt.title('Percentage of missing values:')
plt.xlabel('% of missing data')

try:
    df.drop(columns=['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], inplace=True)
except KeyError:
    print('Features already dropped')
    pass
columns_missing_less_200 = df[df.select_dtypes(exclude=object).columns].loc[:, df.isnull().sum() < 200].columns
for col in columns_missing_less_200:
    df[col] = df[col].fillna(df[col].median())
df[df.select_dtypes(exclude=object).columns].loc[:, df.isnull().any()].isna().sum()
data_corr = df.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
data_corr.loc[data_corr['level_0'] == 'LotFrontage']
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby(['LotArea'])['LotFrontage'].transform('median'))
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby(['1stFlrSF'])['LotFrontage'].transform('median'))
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby(['MSSubClass'])['LotFrontage'].transform('median'))
df[df.select_dtypes(exclude=object).columns].loc[:, df.isnull().any()].isna().sum()
corrMatrix = df.corr()
mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(28, 12))
sns.heatmap(corrMatrix, mask=mask, vmax=0.3, square=False, annot=True)

plt.subplots(figsize=(20, 5))
corrMatrix[target].drop([target]).sort_values().plot(kind='bar')
plt.axhline(y=0.05, color='red', linestyle='--')
plt.axhline(y=-0.05, color='red', linestyle='--')

cor_target = abs(corrMatrix[target])
correlated_features = cor_target[cor_target >= 0.05]
print('Amount of total features: ', len(df.select_dtypes(include=object).columns))
print('Amount of correlated features: ', len(correlated_features))
for i in range(0, df[correlated_features.index].shape[1], 6):
    sns.pairplot(data=df[correlated_features.index], x_vars=df[correlated_features.index].columns[i:i + 6], y_vars=['SalePrice'], kind='reg')
df.loc[(df['BsmtFinSF1'] > 4000) & df[target].notnull(), 'BsmtFinSF1'] = None
df.loc[(df['TotalBsmtSF'] > 4000) & df[target].notnull(), 'TotalBsmtSF'] = None
df.loc[(df['GrLivArea'] > 4000) & (df[target] < 400000) & df[target].notnull(), 'GrLivArea'] = None
df.loc[(df['1stFlrSF'] > 4000) & df[target].notnull(), '1stFlrSF'] = None
df.loc[(df['LotFrontage'] > 200) & df[target].notnull(), 'LotFrontage'] = None
df.loc[(df['LotArea'] > 60000) & df[target].notnull(), 'LotArea'] = None
df.loc[(df['GarageArea'] > 1200) & (df[target] < 400000) & df[target].notnull(), 'GarageArea'] = None
for i in range(0, df[correlated_features.index].shape[1], 6):
    sns.pairplot(data=df[correlated_features.index], x_vars=df[correlated_features.index].columns[i:i + 6], y_vars=['SalePrice'], kind='reg')
df[df.select_dtypes(include=object).columns].loc[:, df.isnull().any()].isna().sum()
for col in df.select_dtypes(include=object).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
df[df.select_dtypes(include=object).columns].loc[:, df.isnull().any()].isna().sum()
import scipy.stats as ss

def cramers_v(confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association"""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    (r, k) = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))
cramers_values = []
obj_columns = df.select_dtypes(include=object).columns
for column in obj_columns:
    confusion_matrix = pd.crosstab(df[column], df[target])
    cramers_values.append(cramers_v(confusion_matrix.values))
cramers_values_df = pd.DataFrame()
(cramers_values_df['feature'], cramers_values_df['value']) = (obj_columns, cramers_values)
plt.subplots(figsize=(12, 10))
plt.barh(cramers_values_df[cramers_values_df['value'] > 0].sort_values(['value'])['feature'], cramers_values_df[cramers_values_df['value'] > 0].sort_values(['value'])['value'])
plt.axvline(x=0.15, color='red', linestyle='--', label='Reference line')
plt.title('Categorical feature correlation with the target')

cramers_values_df = cramers_values_df[cramers_values_df['value'] > 0.15]['feature'].to_list()
len(cramers_values_df)
df = pd.concat([df[cramers_values_df], df[correlated_features.index]], axis=1)
df
dummy_features = df.select_dtypes(include=object).columns
df = pd.concat([df, pd.get_dummies(df[dummy_features], drop_first=True)], axis=1, sort=False)
df.drop(columns=df[dummy_features], inplace=True)
df.tail()
test = df[df[target].isnull()].drop([target], axis=1).copy()
train = df[df[target].notnull()].copy()
train.head(5)
train.dropna(inplace=True)
test.dropna(inplace=True)
y = train[target]
x = train.drop(columns=target)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
(x_train.shape, x_test.shape)
CBR = CatBoostRegressor()
CBR_params = {'n_estimators': [n for n in range(600, 1400, 200)], 'max_depth': [n for n in range(2, 10, 2)], 'random_state': [42], 'learning_rate': [0.05, 0.03]}
CBR_model = GridSearchCV(CBR, param_grid=CBR_params, cv=5, n_jobs=-1)