import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
train_data.info()
train_data.describe()
train_data.head(round(len(train_data) / 2))
train_data.tail(round(len(train_data) / 2))
test_data.info()
test_data.describe()
test_data.head(round(len(test_data) / 2))
test_data.tail(round(len(test_data) / 2))
fig = plt.figure(figsize=(18, 25))
ax = fig.gca()
train_data.hist(ax=ax)
num_cols = train_data.drop('SalePrice', axis=1).select_dtypes('number').columns
count = len(num_cols)
(fig, ax) = plt.subplots(int(count / 2) + (count % 2 > 0), 2, figsize=(18, 200))
for i in range(count):
    x_pos = int(i / 2)
    y_pos = i % 2
    sns.scatterplot(data=train_data, x=num_cols[i], y='SalePrice', ax=ax[x_pos, y_pos])
fig.show()
qual_cond_pivot_table = train_data.pivot_table(values='SalePrice', aggfunc=np.median, index='OverallQual', columns='OverallCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
bsmt_qual_cond_pivot_table = train_data.pivot_table(values='SalePrice', aggfunc=np.median, index='BsmtQual', columns='BsmtCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(bsmt_qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
exter_qual_cond_pivot_table = train_data.pivot_table(values='SalePrice', aggfunc=np.median, index='ExterQual', columns='ExterCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(exter_qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
house_style_type_pivot_table = train_data.pivot_table(values='SalePrice', aggfunc=np.median, index='HouseStyle', columns='BldgType', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(house_style_type_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
train_data['KitchenQual'].value_counts()
object_cols = train_data.select_dtypes('object').columns
count = len(object_cols)
(fig, ax) = plt.subplots(int(count / 2) + (count % 2 > 0), 2, figsize=(18, 200))
for i in range(count):
    x_pos = int(i / 2)
    y_pos = i % 2
    counts = train_data[object_cols[i]].value_counts()
    ax[x_pos, y_pos].pie(counts, labels=counts.index, autopct='%.0f%%')
    ax[x_pos, y_pos].set_title(object_cols[i])
fig.show()
fig = plt.figure(figsize=(18, 200))
object_cols = train_data.select_dtypes('object').columns
count = len(object_cols)
y_count = int(count / 2) + (count % 2 > 0)
x_count = 2
for i in range(count):
    ax = fig.add_subplot(y_count, x_count, i + 1)
    g = sns.boxplot(x=object_cols[i], y='SalePrice', data=train_data, ax=ax)
fig.show()
plt.figure(figsize=(20, 15))
sns.heatmap(data=train_data.corr(), cmap='coolwarm', annot=True, linewidths=0.5)
plt.figure(figsize=(20, 15))
df_corr = train_data.corr()
filtered_corr = df_corr[((df_corr >= 0.5) | (df_corr <= -0.5)) & (df_corr != 1.0)]
sns.heatmap(data=filtered_corr, cmap='coolwarm', annot=True, linewidths=0.5)
train_data.corr()['SalePrice'].sort_values(ascending=False)[:10]
train_data.isna().sum().sort_values(ascending=False)[:20]
test_data.isna().sum().sort_values(ascending=False)[:30]
print(train_data.PoolQC.unique())
print(train_data.MiscFeature.unique())
print(train_data.Alley.unique())
print(train_data.Fence.unique())
print(train_data.FireplaceQu.unique())
columns_to_fill = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in columns_to_fill:
    train_data[col].fillna('NA', inplace=True)
    test_data[col].fillna('NA', inplace=True)
train_data['LotFrontage'].hist()
missed_frontage_data = train_data[train_data.LotFrontage.isnull()][['LotFrontage', 'LotArea', 'LotConfig', 'Neighborhood', 'SaleType', 'SalePrice']]
print(missed_frontage_data.shape)
print(missed_frontage_data.head(125))
missed_frontage_data.LotConfig.unique()
mask = train_data.LotFrontage.isnull() & train_data.LotConfig == 'Inside'
train_data.loc[mask, 'LotFrontage'] = 0
mask = test_data.LotFrontage.isnull() & test_data.LotConfig == 'Inside'
test_data.loc[mask, 'LotFrontage'] = 0
train_data['LotFrontage'].fillna(train_data['LotFrontage'].median(), inplace=True)
test_data['LotFrontage'].fillna(test_data['LotFrontage'].median(), inplace=True)
train_data[['GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType', 'Electrical']].info()
test_data[['MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath', 'Utilities', 'BsmtFinSF1', 'Exterior1st', 'KitchenQual', 'GarageCars', 'GarageArea', 'Exterior2nd', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'SaleType']].info()
numerical_cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
categorical_cols = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'MasVnrType', 'Electrical', 'MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'KitchenQual', 'Exterior2nd', 'SaleType']
for cat_col in categorical_cols:
    train_data[cat_col].fillna(train_data[cat_col].mode()[0], inplace=True)
    test_data[cat_col].fillna(test_data[cat_col].mode()[0], inplace=True)
for num_col in numerical_cols:
    train_data[num_col].fillna(train_data[num_col].median(), inplace=True)
    test_data[num_col].fillna(test_data[num_col].median(), inplace=True)
print(test_data.isna().sum().sort_values(ascending=False)[:5])
print(train_data.isna().sum().sort_values(ascending=False)[:5])
train_data['OverallQualPlusCond'] = train_data['OverallQual'] + train_data['OverallCond']
test_data['OverallQualPlusCond'] = test_data['OverallQual'] + test_data['OverallCond']
train_data['OverallQualMultCond'] = train_data['OverallQual'] * train_data['OverallCond']
test_data['OverallQualMultCond'] = test_data['OverallQual'] * test_data['OverallCond']
print(train_data[['OverallQual', 'OverallCond', 'OverallQualPlusCond', 'OverallQualMultCond']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=train_data, x='OverallQualPlusCond', y='SalePrice', ax=ax[0])
sns.scatterplot(data=train_data, x='OverallQualMultCond', y='SalePrice', ax=ax[1])
fig.show()
train_data['HasRemod'] = train_data.YearBuilt != train_data.YearRemodAdd
train_data['HasRemod'] = train_data['HasRemod'].astype(int)
print(train_data[['YearBuilt', 'YearRemodAdd', 'HasRemod']].head())
test_data['HasRemod'] = test_data.YearBuilt != test_data.YearRemodAdd
test_data['HasRemod'] = test_data['HasRemod'].astype(int)
remode_median = train_data.loc[train_data['HasRemod'] == 1, 'SalePrice'].median() / 1000
no_remode_median = train_data[train_data['HasRemod'] == 0]['SalePrice'].median() / 1000
plt.figure(figsize=(8, 5))
plt.boxplot(x=[train_data[train_data['HasRemod'] == 1]['SalePrice'], train_data[train_data['HasRemod'] == 0]['SalePrice']], labels=[f'Remod ({remode_median})', f'No Remod ({no_remode_median})'])

train_data['ExterQualCond'] = train_data['ExterQual'] + train_data['ExterCond']
test_data['ExterQualCond'] = test_data['ExterQual'] + test_data['ExterCond']
print(train_data[['ExterQual', 'ExterCond', 'ExterQualCond']].head())
train_data['BsmtQualCond'] = train_data['BsmtQual'] + train_data['BsmtCond']
test_data['BsmtQualCond'] = test_data['BsmtQual'] + test_data['BsmtCond']
print(train_data[['BsmtQual', 'BsmtCond', 'BsmtQualCond']].head())
train_data['GarageQualCond'] = train_data['GarageQual'] + train_data['GarageCond']
test_data['GarageQualCond'] = test_data['GarageQual'] + test_data['GarageCond']
print(train_data[['GarageQual', 'GarageCond', 'GarageQualCond']].head())
(fig, ax) = plt.subplots(2, 2, figsize=(20, 8))
fig.tight_layout()
sns.boxplot(x='ExterQualCond', y='SalePrice', data=train_data, ax=ax[0, 0])
sns.boxplot(x='BsmtQualCond', y='SalePrice', data=train_data, ax=ax[0, 1])
sns.boxplot(x='GarageQualCond', y='SalePrice', data=train_data, ax=ax[1, 0])
fig.show()
train_data['BsmtRatio1'] = train_data['BsmtUnfSF'] / (train_data['TotalBsmtSF'] + 1)
test_data['BsmtRatio1'] = test_data['BsmtUnfSF'] / (test_data['TotalBsmtSF'] + 1)
train_data['BsmtRatio2'] = (train_data['BsmtFinSF1'] + train_data['BsmtFinSF2']) / (train_data['TotalBsmtSF'] + 1)
test_data['BsmtRatio2'] = (test_data['BsmtFinSF1'] + test_data['BsmtFinSF2']) / (test_data['TotalBsmtSF'] + 1)
print(train_data[['BsmtUnfSF', 'TotalBsmtSF', 'BsmtRatio1', 'BsmtRatio2']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=train_data, x='BsmtRatio1', y='SalePrice', ax=ax[0])
sns.scatterplot(data=train_data, x='BsmtRatio2', y='SalePrice', ax=ax[1])
fig.show()
train_data['HasWoodDeck'] = train_data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
test_data['HasWoodDeck'] = test_data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train_data['HasOpenPorch'] = train_data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
test_data['HasOpenPorch'] = test_data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
print(train_data[['WoodDeckSF', 'HasWoodDeck', 'OpenPorchSF', 'HasOpenPorch', 'SalePrice']].head())
wood_deck_median = train_data.loc[train_data['HasWoodDeck'] == 1, 'SalePrice'].median() / 1000
no_wood_deck_median = train_data[train_data['HasWoodDeck'] == 0]['SalePrice'].median() / 1000
(fig, ax) = plt.subplots(2, 2, figsize=(18, 8))
ax[0, 0].set_title('HasWoodDeck')
ax[0, 0].boxplot(x=[train_data[train_data['HasWoodDeck'] == 1]['SalePrice'], train_data[train_data['HasWoodDeck'] == 0]['SalePrice']], labels=[f'WoodDeck ({wood_deck_median})', f'No WoodDeck ({no_remode_median})'])
open_porch_median = train_data.loc[train_data['HasOpenPorch'] == 1, 'SalePrice'].median() / 1000
no_open_porch_median = train_data[train_data['HasOpenPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('HasOpenPorch')
ax[0, 1].boxplot(x=[train_data[train_data['HasOpenPorch'] == 1]['SalePrice'], train_data[train_data['HasOpenPorch'] == 0]['SalePrice']], labels=[f'OpenPorch ({open_porch_median})', f'No OpenPorch ({no_open_porch_median})'])
ax[1, 0].set_title('HasWoodDeck')
ax[1, 0].hist(train_data['HasWoodDeck'])
ax[1, 1].set_title('HasOpenPorch')
ax[1, 1].hist(train_data['HasOpenPorch'])
fig.show()
(fig, ax) = plt.subplots(4, 2, figsize=(18, 8))
fig.tight_layout()
train_data['HasSecondFloor'] = train_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test_data['HasSecondFloor'] = test_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
second_floor_median = train_data.loc[train_data['HasSecondFloor'] == 1, 'SalePrice'].median() / 1000
no_second_floor_median = train_data[train_data['HasSecondFloor'] == 0]['SalePrice'].median() / 1000
ax[0, 0].set_title('SecondFloor')
ax[0, 0].boxplot(x=[train_data[train_data['HasSecondFloor'] == 1]['SalePrice'], train_data[train_data['HasSecondFloor'] == 0]['SalePrice']], labels=[f'SecondFloor ({second_floor_median})', f'No SecondFloor ({no_second_floor_median})'])
ax[1, 0].set_title('SecondFloor')
ax[1, 0].hist(train_data['HasSecondFloor'])
train_data['NoGrdBath'] = np.logical_and(train_data['FullBath'] == 0, train_data['HalfBath'] == 0).astype(int)
test_data['NoGrdBath'] = np.logical_and(test_data['FullBath'] == 0, test_data['HalfBath'] == 0).astype(int)
grd_bath_median = train_data.loc[train_data['NoGrdBath'] == 1, 'SalePrice'].median() / 1000
no_grd_bath_median = train_data[train_data['NoGrdBath'] == 0]['SalePrice'].median() / 1000
ax[2, 0].set_title('GrdBath')
ax[2, 0].boxplot(x=[train_data[train_data['NoGrdBath'] == 1]['SalePrice'], train_data[train_data['NoGrdBath'] == 0]['SalePrice']], labels=[f'GrdBath ({grd_bath_median})', f'No GrdBath ({no_grd_bath_median})'])
ax[3, 0].set_title('GrdBath')
ax[3, 0].hist(train_data['NoGrdBath'])
train_data['NoBsmtBath'] = pd.Series(np.logical_and(train_data['BsmtFullBath'] == 0, train_data['BsmtHalfBath'] == 0)).astype(int)
test_data['NoBsmtBath'] = pd.Series(np.logical_and(test_data['BsmtFullBath'] == 0, test_data['BsmtHalfBath'] == 0)).astype(int)
bsmt_bath_median = train_data.loc[train_data['NoBsmtBath'] == 1, 'SalePrice'].median() / 1000
no_bsmt_bath_median = train_data[train_data['NoBsmtBath'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('BsmtBath')
ax[0, 1].boxplot(x=[train_data[train_data['NoBsmtBath'] == 1]['SalePrice'], train_data[train_data['NoBsmtBath'] == 0]['SalePrice']], labels=[f'BsmtBath ({bsmt_bath_median})', f'No BsmtBath ({no_bsmt_bath_median})'])
ax[1, 1].set_title('BsmtBath')
ax[1, 1].hist(train_data['NoBsmtBath'])
train_data['NoBath'] = np.logical_and(train_data['NoBsmtBath'] == 1, train_data['NoGrdBath'] == 1).astype(int)
test_data['NoBath'] = np.logical_and(test_data['NoBsmtBath'] == 1, test_data['NoGrdBath'] == 1).astype(int)
bath_median = train_data.loc[train_data['NoBath'] == 1, 'SalePrice'].median() / 1000
no_bath_median = train_data[train_data['NoBath'] == 0]['SalePrice'].median() / 1000
ax[2, 1].set_title('Bath')
ax[2, 1].boxplot(x=[train_data[train_data['NoBath'] == 1]['SalePrice'], train_data[train_data['NoBath'] == 0]['SalePrice']], labels=[f'BsmtBath ({bath_median})', f'No BsmtBath ({no_bath_median})'])
ax[3, 1].set_title('Bath')
ax[3, 1].hist(train_data['NoBath'])
fig.show()
(fig, ax) = plt.subplots(4, 2, figsize=(18, 8))
fig.tight_layout()
train_data['HasEnclosedPorch'] = train_data['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
test_data['HasEnclosedPorch'] = test_data['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
enclosed_porch_median = train_data.loc[train_data['HasEnclosedPorch'] == 1, 'SalePrice'].median() / 1000
no_enclosed_porch_median = train_data[train_data['HasEnclosedPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 0].set_title('Enclosed Porch')
ax[0, 0].boxplot(x=[train_data[train_data['HasEnclosedPorch'] == 1]['SalePrice'], train_data[train_data['HasEnclosedPorch'] == 0]['SalePrice']], labels=[f'Enclosed Porch ({enclosed_porch_median})', f'No Enclosed Porch ({no_enclosed_porch_median})'])
ax[1, 0].set_title('Enclosed Porch')
ax[1, 0].hist(train_data['HasEnclosedPorch'])
train_data['Has3SsnPorch'] = train_data['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
test_data['Has3SsnPorch'] = test_data['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
three_ssn_median = train_data.loc[train_data['Has3SsnPorch'] == 1, 'SalePrice'].median() / 1000
no_three_ssn_median = train_data[train_data['Has3SsnPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('3 Seasons Porch')
ax[0, 1].boxplot(x=[train_data[train_data['Has3SsnPorch'] == 1]['SalePrice'], train_data[train_data['Has3SsnPorch'] == 0]['SalePrice']], labels=[f'3 Seasons Porch ({three_ssn_median})', f'No 3 Seasons Porch ({no_three_ssn_median})'])
ax[1, 1].set_title('3 Seasons Porch')
ax[1, 1].hist(train_data['Has3SsnPorch'])
train_data['HasScreenPorch'] = train_data['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
test_data['HasScreenPorch'] = test_data['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
scrn_porch_median = train_data.loc[train_data['HasScreenPorch'] == 1, 'SalePrice'].median() / 1000
no_scrn_porch_median = train_data[train_data['HasScreenPorch'] == 0]['SalePrice'].median() / 1000
ax[2, 0].set_title('Screen Porch')
ax[2, 0].boxplot(x=[train_data[train_data['HasScreenPorch'] == 1]['SalePrice'], train_data[train_data['HasScreenPorch'] == 0]['SalePrice']], labels=[f'GrdBath ({scrn_porch_median})', f'No GrdBath ({no_scrn_porch_median})'])
ax[3, 0].set_title('Screen Porch')
ax[3, 0].hist(train_data['HasScreenPorch'])
train_data['HasPorch'] = np.logical_or(np.logical_or(train_data['HasOpenPorch'] == 1, train_data['HasEnclosedPorch'] == 1), np.logical_or(train_data['Has3SsnPorch'] == 1, train_data['HasScreenPorch'] == 1)).astype(int)
test_data['HasPorch'] = np.logical_or(np.logical_or(test_data['HasOpenPorch'] == 1, test_data['HasEnclosedPorch'] == 1), np.logical_or(test_data['Has3SsnPorch'] == 1, test_data['HasScreenPorch'] == 1)).astype(int)
porch_median = train_data.loc[train_data['HasPorch'] == 1, 'SalePrice'].median() / 1000
no_porch_median = train_data[train_data['HasPorch'] == 0]['SalePrice'].median() / 1000
ax[2, 1].set_title('Porch')
ax[2, 1].boxplot(x=[train_data[train_data['HasPorch'] == 1]['SalePrice'], train_data[train_data['HasPorch'] == 0]['SalePrice']], labels=[f'Porch ({porch_median})', f'No Porch ({no_porch_median})'])
ax[3, 1].set_title('Porch')
ax[3, 1].hist(train_data['HasPorch'])
print(train_data[['OpenPorchSF', 'HasOpenPorch', 'EnclosedPorch', 'HasEnclosedPorch', '3SsnPorch', 'Has3SsnPorch', 'ScreenPorch', 'HasScreenPorch', 'HasPorch', 'SalePrice']].head())
fig.show()
train_data['LotArea'].describe()
train_data['FrontageRatio'] = train_data['LotFrontage'] / train_data['LotArea']
test_data['FrontageRatio'] = test_data['LotFrontage'] / test_data['LotArea']
print(train_data[['LotFrontage', 'LotArea', 'FrontageRatio', 'SalePrice']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=train_data, x='BsmtRatio1', y='SalePrice', ax=ax[0])
ax[1].hist(train_data['FrontageRatio'])
fig.show()
print(train_data.YrSold.unique())
print(test_data.YrSold.unique())
train_data['YrSoldLog'] = np.log(train_data['YrSold'] - 2000)
train_data['YrSoldCube'] = train_data['YrSold'] ** 3
train_data['YrSoldSqrt'] = train_data['YrSold'] ** 0.5
test_data['YrSoldLog'] = np.log(test_data['YrSold'] - 2000)
test_data['YrSoldCube'] = test_data['YrSold'] ** 3
test_data['YrSoldSqrt'] = test_data['YrSold'] ** 0.5
print(train_data[['YrSold', 'YrSoldLog', 'YrSoldCube', 'YrSoldSqrt']].head())
(fig, ax) = plt.subplots(1, 4, figsize=(18, 4))
ax[0].set_title('YrSold')
ax[0].hist(train_data['YrSold'])
ax[1].set_title('YrSoldCube')
ax[1].hist(train_data['YrSoldCube'])
ax[2].set_title('YrSoldSqrt')
ax[2].hist(train_data['YrSoldSqrt'])
ax[3].set_title('YrSoldLog')
ax[3].hist(train_data['YrSoldLog'])
fig.show()
train_data['MSSubClassCbrt'] = train_data['MSSubClass'] ** (1 / 3)
train_data['MSSubClassSquare'] = train_data['MSSubClass'] ** 2
train_data['MSSubClassLog'] = np.log(train_data['MSSubClass'] + 1)
test_data['MSSubClassCbrt'] = test_data['MSSubClass'] ** (1 / 3)
test_data['MSSubClassSquare'] = test_data['MSSubClass'] ** 2
test_data['MSSubClassLog'] = np.log(test_data['MSSubClass'] + 1)
train_data['MoSoldSqrt'] = train_data['MoSold'] ** (1 / 2)
train_data['MoSoldCubic'] = train_data['MoSold'] ** 3
train_data['MoSoldLog'] = np.log(train_data['MoSold'] + 1)
test_data['MoSoldSqrt'] = test_data['MoSold'] ** (1 / 2)
test_data['MoSoldCubic'] = test_data['MoSold'] ** 3
test_data['MoSoldLog'] = np.log(test_data['MoSold'] + 1)
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('MSSubClass')
ax[0, 0].hist(train_data['MSSubClass'])
ax[0, 1].set_title('MSSubClassCbrt')
ax[0, 1].hist(train_data['MSSubClassCbrt'])
ax[0, 2].set_title('MSSubClassSquare')
ax[0, 2].hist(train_data['MSSubClassSquare'])
ax[0, 3].set_title('MSSubClassLog')
ax[0, 3].hist(train_data['MSSubClassLog'])
ax[1, 0].set_title('MoSold')
ax[1, 0].hist(train_data['MoSold'])
ax[1, 1].set_title('MoSoldSqrt')
ax[1, 1].hist(train_data['MoSoldSqrt'])
ax[1, 2].set_title('MoSoldCubic')
ax[1, 2].hist(train_data['MoSoldCubic'])
ax[1, 3].set_title('MoSoldLog')
ax[1, 3].hist(train_data['MoSoldLog'])
fig.show()
train_data['GarageYrBuiltCbrt'] = train_data['GarageYrBlt'] ** (1 / 3)
train_data['GarageYrBuiltSquare'] = train_data['GarageYrBlt'] ** 2
train_data['GarageYrBuiltLog'] = np.log(train_data['GarageYrBlt'] - 1800)
test_data['GarageYrBuiltCbrt'] = test_data['GarageYrBlt'] ** (1 / 3)
test_data['GarageYrBuiltSquare'] = test_data['GarageYrBlt'] ** 2
test_data['GarageYrBuiltLog'] = np.log(test_data['GarageYrBlt'] - 1800)
train_data['YearBuiltSqrt'] = train_data['YearBuilt'] ** (1 / 2)
train_data['YearBuiltCubic'] = train_data['YearBuilt'] ** 3
train_data['YearBuiltLog'] = np.log(train_data['YearBuilt'] - 1700)
test_data['YearBuiltSqrt'] = test_data['YearBuilt'] ** (1 / 2)
test_data['YearBuiltCubic'] = test_data['YearBuilt'] ** 3
test_data['YearBuiltLog'] = np.log(test_data['YearBuilt'] - 1700)
print(train_data[['YearBuilt', 'YearBuiltSqrt', 'YearBuiltCubic', 'YearBuiltLog', 'GarageYrBlt', 'GarageYrBuiltCbrt', 'GarageYrBuiltSquare', 'GarageYrBuiltLog']].head(500))
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('YearBuilt')
ax[0, 0].hist(train_data['YearBuilt'])
ax[0, 1].set_title('YearBuiltSqrt')
ax[0, 1].hist(train_data['YearBuiltSqrt'])
ax[0, 2].set_title('YearBuiltCubic')
ax[0, 2].hist(train_data['YearBuiltCubic'])
ax[0, 3].set_title('YearBuiltLog')
ax[0, 3].hist(train_data['YearBuiltLog'])
ax[1, 0].set_title('GarageYrBlt')
ax[1, 0].hist(train_data['GarageYrBlt'])
ax[1, 1].set_title('GarageYrBuiltCbrt')
ax[1, 1].hist(train_data['GarageYrBuiltCbrt'])
ax[1, 2].set_title('GarageYrBuiltSquare')
ax[1, 2].hist(train_data['GarageYrBuiltSquare'])
ax[1, 3].set_title('GarageYrBuiltLog')
ax[1, 3].hist(train_data['GarageYrBuiltLog'])
fig.show()
train_data['GarageAreaLog'] = np.log(train_data['GarageArea'] + 1)
test_data['GarageAreaLog'] = np.log(test_data['GarageArea'] + 1)
train_data['GarageAreaSquare'] = train_data['GarageArea'] ** 2
test_data['GarageAreaSquare'] = test_data['GarageArea'] ** 2
train_data['GarageAreaCbrt'] = train_data['GarageArea'] ** (1 / 3)
test_data['GarageAreaCbrt'] = test_data['GarageArea'] ** (1 / 3)
train_data['BsmtUnfSFLog'] = np.log(train_data['BsmtUnfSF'] + 5)
test_data['BsmtUnfSFLog'] = np.log(test_data['BsmtUnfSF'] + 5)
train_data['BsmtUnfSFCube'] = train_data['BsmtUnfSF'] ** 3
test_data['BsmtUnfSFCube'] = test_data['BsmtUnfSF'] ** 3
train_data['BsmtUnfSFSqrt'] = train_data['BsmtUnfSF'] ** (1 / 2)
test_data['BsmtUnfSFSqrt'] = test_data['BsmtUnfSF'] ** (1 / 2)
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('GarageArea')
ax[0, 0].hist(train_data['GarageArea'])
ax[0, 1].set_title('GarageAreaLog')
ax[0, 1].hist(train_data['GarageAreaLog'])
ax[0, 2].set_title('GarageAreaSquare')
ax[0, 2].hist(train_data['GarageAreaSquare'])
ax[0, 3].set_title('GarageAreaCbrt')
ax[0, 3].hist(train_data['GarageAreaCbrt'])
ax[1, 0].set_title('BsmtUnfSF')
ax[1, 0].hist(train_data['BsmtUnfSF'])
ax[1, 1].set_title('BsmtUnfSFLog')
ax[1, 1].hist(train_data['BsmtUnfSFLog'])
ax[1, 2].set_title('GarageYrBuiltSquare')
ax[1, 2].hist(train_data['GarageYrBuiltSquare'])
ax[1, 3].set_title('BsmtUnfSFSqrt')
ax[1, 3].hist(train_data['BsmtUnfSFSqrt'])
fig.show()
train_data['OverallQualSquare'] = train_data['OverallQual'] ** 2
test_data['OverallQualSquare'] = test_data['OverallQual'] ** 2
train_data['OverallQualCube'] = train_data['OverallQual'] ** 3
test_data['OverallQualCube'] = test_data['OverallQual'] ** 3
train_data['OverallQualPlusCondSquare'] = train_data['OverallQualPlusCond'] ** 2
test_data['OverallQualPlusCondSquare'] = test_data['OverallQualPlusCond'] ** 2
train_data['OverallQualPlusCondCube'] = train_data['OverallQualPlusCond'] ** 3
test_data['OverallQualPlusCondCube'] = test_data['OverallQualPlusCond'] ** 3
(fig, ax) = plt.subplots(2, 3, figsize=(12, 4))
fig.tight_layout()
ax[0, 0].set_title('OverallQual')
ax[0, 0].hist(train_data['OverallQual'])
ax[0, 1].set_title('OverallQualSquare')
ax[0, 1].hist(train_data['OverallQualSquare'])
ax[0, 2].set_title('OverallQualCube')
ax[0, 2].hist(train_data['OverallQualCube'])
ax[1, 0].set_title('OverallQualPlusCond')
ax[1, 0].hist(train_data['OverallQualPlusCond'])
ax[1, 1].set_title('OverallQualPlusCondSquare')
ax[1, 1].hist(train_data['OverallQualPlusCondSquare'])
ax[1, 2].set_title('OverallQualPlusCondCube')
ax[1, 2].hist(train_data['OverallQualPlusCondCube'])
fig.show()

def handle_outliers(df: pd.DataFrame, col: str):
    res_df = df.copy()
    q1 = np.nanpercentile(res_df[col], 25)
    q3 = np.nanpercentile(res_df[col], 75)
    IRQ = q3 - q1
    min_value = q1 - 1.5 * IRQ
    max_value = q3 + 1.5 * IRQ
    median = res_df[col].median()
    print_log = False
    if print_log:
        print(f'\ncol = {col}; len={len(res_df[col])}')
        print(f'median ={median}, q1={q1}, q1={q3}, IRQ={IRQ}')
        print(f'max={max_value}, min={min_value}')
        print(f'top outliers count = {len(res_df.loc[res_df[col] > max_value, col])}; ')
        print(f'bottom outliers count = {len(res_df.loc[res_df[col] < min_value, col])}; \n')
    res_df.loc[res_df[col] > max_value, col] = max_value
    res_df.loc[res_df[col] < min_value, col] = min_value
    return res_df

def handle_multiple_outliers(df: pd.DataFrame, cols: list):
    new_df = df.copy()
    for col in cols:
        new_df = handle_outliers(new_df, col)
    return new_df
numeric_columns = test_data.select_dtypes('number').columns
train_data = handle_multiple_outliers(train_data, numeric_columns)
test_data = handle_multiple_outliers(test_data, numeric_columns)
for col in numeric_columns:
    plt.figure(figsize=(12, 1))
    ax = sns.boxplot(x=train_data[col])
train_data['MSSubClass'] = train_data['MSSubClass'].astype(str)
test_data['MSSubClass'] = test_data['MSSubClass'].astype(str)
train_data['MoSold'] = train_data['MoSold'].astype(str)
test_data['MoSold'] = test_data['MoSold'].astype(str)
train_data['YrSold'] = train_data['YrSold'].astype(str)
test_data['YrSold'] = test_data['YrSold'].astype(str)
train_data['OverallQualPlusCond'] = train_data['OverallQualPlusCond'].astype(str)
test_data['OverallQualPlusCond'] = test_data['OverallQualPlusCond'].astype(str)
train_data['OverallQual'] = train_data['OverallQual'].astype(str)
test_data['OverallQual'] = test_data['OverallQual'].astype(str)
train_data['OverallCond'] = train_data['OverallCond'].astype(str)
test_data['OverallCond'] = test_data['OverallCond'].astype(str)
print(train_data['OverallQual'].describe())
print(train_data['MoSold'].describe())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = test_data.select_dtypes('number').columns
X_num_train_scaled = scaler.fit_transform(train_data[numeric_columns])
print(X_num_train_scaled[:2])
from sklearn.linear_model import Lasso
X_lasso = X_num_train_scaled
y_lasso = train_data['SalePrice']
lasso = Lasso(tol=0.01, positive=True)