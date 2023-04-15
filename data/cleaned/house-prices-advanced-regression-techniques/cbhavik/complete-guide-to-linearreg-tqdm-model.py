import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 100)
train0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(train0.shape, test0.shape)
train0.tail()
test0.head()
plt.figure(figsize=(24, 13))
d = train0.drop('Id', axis=1)
corr = d.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', mask=mask)
target = train0['SalePrice']
test_ids = test0['Id']
train1 = train0.drop(['Id', 'SalePrice'], axis=1)
test1 = test0.drop('Id', axis=1)
data1 = pd.concat([train1, test1], axis=0).reset_index(drop=True)
data1
target
plt.figure(figsize=(24, 13))
d = data1
corr = d.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', mask=mask)
data2 = data1.copy()
data2 = data2.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})
quantitative = [f for f in data2.columns if data2.dtypes[f] != 'O']
qualitative = [f for f in data2.columns if data2.dtypes[f] == 'O']
(len(quantitative), len(qualitative))
plt.figure(figsize=(10, 8))
missing = data2.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(ascending=False, inplace=True)
missing.plot.bar()
missing_df = missing.to_frame().reset_index()
missing_df.columns = ['Feature', 'Num_of_missing_values']
missing_df['Percent_missing_value'] = missing_df['Num_of_missing_values'] * 100 / len(data2)
missing_df
data2.select_dtypes('O').loc[:, data2.isna().sum() > 0].columns
for feature in ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    data2[feature] = data2[feature].fillna('None')
for feature in ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'SaleType']:
    data2[feature] = data2[feature].fillna('Missing')
data2.select_dtypes('O').loc[:, data2.isna().sum() > 0].columns
import scipy.stats as st
test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(data2[quantitative])
normal = normal.apply(test_normality)
print(not normal.any())
f = pd.melt(data2, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
data3 = data2.copy()
numeric_na_features = data3.select_dtypes(np.number).loc[:, data3.isna().sum() > 0].columns
numeric_na_features
for feature in numeric_na_features:
    df = data3.copy()
    df[feature] = np.log(df[feature])
    df.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(f"{feature}'s Outliers'")


def knn_impute(df, na_target):
    """This will take dataframe & column with missing value as i/p and predict the missing values in that column using KNN."""
    from sklearn.neighbors import KNeighborsRegressor
    df = df.copy()
    numeric_df = df.select_dtypes(np.number)
    non_na_columns = numeric_df.loc[:, numeric_df.isna().sum() == 0].columns
    y_train = numeric_df.loc[numeric_df[na_target].isna() == False, na_target]
    X_train = numeric_df.loc[numeric_df[na_target].isna() == False, non_na_columns]
    X_test = numeric_df.loc[numeric_df[na_target].isna() == True, non_na_columns]
    y_test = numeric_df.loc[numeric_df[na_target].isna() == True, na_target]
    knn = KNeighborsRegressor()