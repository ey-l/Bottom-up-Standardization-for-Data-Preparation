import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

color = sns.color_palette()
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train_ID = train['Id']
test_ID = test['Id']
plt.figure(figsize=(24, 13))
d = train.drop('Id', axis=1)
corr = d.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', mask=mask)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
(fig, ax) = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
(fig, ax) = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

numeric_na_features = train.select_dtypes(np.number).loc[:, train.isna().sum() > 0].columns
numeric_na_features
for feature in numeric_na_features:
    df = train.copy()
    df[feature] = np.log(df[feature])
    df.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(f"{feature}'s Outliers'")

sns.distplot(train['SalePrice'], fit=norm)