import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input1.info()
_input1.describe()
_input1.head(round(len(_input1) / 2))
_input1.tail(round(len(_input1) / 2))
_input0.info()
_input0.describe()
_input0.head(round(len(_input0) / 2))
_input0.tail(round(len(_input0) / 2))
fig = plt.figure(figsize=(18, 25))
ax = fig.gca()
_input1.hist(ax=ax)
num_cols = _input1.drop('SalePrice', axis=1).select_dtypes('number').columns
count = len(num_cols)
(fig, ax) = plt.subplots(int(count / 2) + (count % 2 > 0), 2, figsize=(18, 200))
for i in range(count):
    x_pos = int(i / 2)
    y_pos = i % 2
    sns.scatterplot(data=_input1, x=num_cols[i], y='SalePrice', ax=ax[x_pos, y_pos])
fig.show()
qual_cond_pivot_table = _input1.pivot_table(values='SalePrice', aggfunc=np.median, index='OverallQual', columns='OverallCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
bsmt_qual_cond_pivot_table = _input1.pivot_table(values='SalePrice', aggfunc=np.median, index='BsmtQual', columns='BsmtCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(bsmt_qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
exter_qual_cond_pivot_table = _input1.pivot_table(values='SalePrice', aggfunc=np.median, index='ExterQual', columns='ExterCond', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(exter_qual_cond_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
house_style_type_pivot_table = _input1.pivot_table(values='SalePrice', aggfunc=np.median, index='HouseStyle', columns='BldgType', fill_value=0, margins=True)
fig = plt.figure(figsize=(12, 8))
sns.heatmap(house_style_type_pivot_table, annot=True, square=True, linewidths=1, cmap='Greens', fmt='g')
_input1['KitchenQual'].value_counts()
object_cols = _input1.select_dtypes('object').columns
count = len(object_cols)
(fig, ax) = plt.subplots(int(count / 2) + (count % 2 > 0), 2, figsize=(18, 200))
for i in range(count):
    x_pos = int(i / 2)
    y_pos = i % 2
    counts = _input1[object_cols[i]].value_counts()
    ax[x_pos, y_pos].pie(counts, labels=counts.index, autopct='%.0f%%')
    ax[x_pos, y_pos].set_title(object_cols[i])
fig.show()
fig = plt.figure(figsize=(18, 200))
object_cols = _input1.select_dtypes('object').columns
count = len(object_cols)
y_count = int(count / 2) + (count % 2 > 0)
x_count = 2
for i in range(count):
    ax = fig.add_subplot(y_count, x_count, i + 1)
    g = sns.boxplot(x=object_cols[i], y='SalePrice', data=_input1, ax=ax)
fig.show()
plt.figure(figsize=(20, 15))
sns.heatmap(data=_input1.corr(), cmap='coolwarm', annot=True, linewidths=0.5)
plt.figure(figsize=(20, 15))
df_corr = _input1.corr()
filtered_corr = df_corr[((df_corr >= 0.5) | (df_corr <= -0.5)) & (df_corr != 1.0)]
sns.heatmap(data=filtered_corr, cmap='coolwarm', annot=True, linewidths=0.5)
_input1.corr()['SalePrice'].sort_values(ascending=False)[:10]
_input1.isna().sum().sort_values(ascending=False)[:20]
_input0.isna().sum().sort_values(ascending=False)[:30]
print(_input1.PoolQC.unique())
print(_input1.MiscFeature.unique())
print(_input1.Alley.unique())
print(_input1.Fence.unique())
print(_input1.FireplaceQu.unique())
columns_to_fill = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in columns_to_fill:
    _input1[col] = _input1[col].fillna('NA', inplace=False)
    _input0[col] = _input0[col].fillna('NA', inplace=False)
_input1['LotFrontage'].hist()
missed_frontage_data = _input1[_input1.LotFrontage.isnull()][['LotFrontage', 'LotArea', 'LotConfig', 'Neighborhood', 'SaleType', 'SalePrice']]
print(missed_frontage_data.shape)
print(missed_frontage_data.head(125))
missed_frontage_data.LotConfig.unique()
mask = _input1.LotFrontage.isnull() & _input1.LotConfig == 'Inside'
_input1.loc[mask, 'LotFrontage'] = 0
mask = _input0.LotFrontage.isnull() & _input0.LotConfig == 'Inside'
_input0.loc[mask, 'LotFrontage'] = 0
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median(), inplace=False)
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].median(), inplace=False)
_input1[['GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType', 'Electrical']].info()
_input0[['MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath', 'Utilities', 'BsmtFinSF1', 'Exterior1st', 'KitchenQual', 'GarageCars', 'GarageArea', 'Exterior2nd', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'SaleType']].info()
numerical_cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
categorical_cols = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'MasVnrType', 'Electrical', 'MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'KitchenQual', 'Exterior2nd', 'SaleType']
for cat_col in categorical_cols:
    _input1[cat_col] = _input1[cat_col].fillna(_input1[cat_col].mode()[0], inplace=False)
    _input0[cat_col] = _input0[cat_col].fillna(_input0[cat_col].mode()[0], inplace=False)
for num_col in numerical_cols:
    _input1[num_col] = _input1[num_col].fillna(_input1[num_col].median(), inplace=False)
    _input0[num_col] = _input0[num_col].fillna(_input0[num_col].median(), inplace=False)
print(_input0.isna().sum().sort_values(ascending=False)[:5])
print(_input1.isna().sum().sort_values(ascending=False)[:5])
_input1['OverallQualPlusCond'] = _input1['OverallQual'] + _input1['OverallCond']
_input0['OverallQualPlusCond'] = _input0['OverallQual'] + _input0['OverallCond']
_input1['OverallQualMultCond'] = _input1['OverallQual'] * _input1['OverallCond']
_input0['OverallQualMultCond'] = _input0['OverallQual'] * _input0['OverallCond']
print(_input1[['OverallQual', 'OverallCond', 'OverallQualPlusCond', 'OverallQualMultCond']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=_input1, x='OverallQualPlusCond', y='SalePrice', ax=ax[0])
sns.scatterplot(data=_input1, x='OverallQualMultCond', y='SalePrice', ax=ax[1])
fig.show()
_input1['HasRemod'] = _input1.YearBuilt != _input1.YearRemodAdd
_input1['HasRemod'] = _input1['HasRemod'].astype(int)
print(_input1[['YearBuilt', 'YearRemodAdd', 'HasRemod']].head())
_input0['HasRemod'] = _input0.YearBuilt != _input0.YearRemodAdd
_input0['HasRemod'] = _input0['HasRemod'].astype(int)
remode_median = _input1.loc[_input1['HasRemod'] == 1, 'SalePrice'].median() / 1000
no_remode_median = _input1[_input1['HasRemod'] == 0]['SalePrice'].median() / 1000
plt.figure(figsize=(8, 5))
plt.boxplot(x=[_input1[_input1['HasRemod'] == 1]['SalePrice'], _input1[_input1['HasRemod'] == 0]['SalePrice']], labels=[f'Remod ({remode_median})', f'No Remod ({no_remode_median})'])
_input1['ExterQualCond'] = _input1['ExterQual'] + _input1['ExterCond']
_input0['ExterQualCond'] = _input0['ExterQual'] + _input0['ExterCond']
print(_input1[['ExterQual', 'ExterCond', 'ExterQualCond']].head())
_input1['BsmtQualCond'] = _input1['BsmtQual'] + _input1['BsmtCond']
_input0['BsmtQualCond'] = _input0['BsmtQual'] + _input0['BsmtCond']
print(_input1[['BsmtQual', 'BsmtCond', 'BsmtQualCond']].head())
_input1['GarageQualCond'] = _input1['GarageQual'] + _input1['GarageCond']
_input0['GarageQualCond'] = _input0['GarageQual'] + _input0['GarageCond']
print(_input1[['GarageQual', 'GarageCond', 'GarageQualCond']].head())
(fig, ax) = plt.subplots(2, 2, figsize=(20, 8))
fig.tight_layout()
sns.boxplot(x='ExterQualCond', y='SalePrice', data=_input1, ax=ax[0, 0])
sns.boxplot(x='BsmtQualCond', y='SalePrice', data=_input1, ax=ax[0, 1])
sns.boxplot(x='GarageQualCond', y='SalePrice', data=_input1, ax=ax[1, 0])
fig.show()
_input1['BsmtRatio1'] = _input1['BsmtUnfSF'] / (_input1['TotalBsmtSF'] + 1)
_input0['BsmtRatio1'] = _input0['BsmtUnfSF'] / (_input0['TotalBsmtSF'] + 1)
_input1['BsmtRatio2'] = (_input1['BsmtFinSF1'] + _input1['BsmtFinSF2']) / (_input1['TotalBsmtSF'] + 1)
_input0['BsmtRatio2'] = (_input0['BsmtFinSF1'] + _input0['BsmtFinSF2']) / (_input0['TotalBsmtSF'] + 1)
print(_input1[['BsmtUnfSF', 'TotalBsmtSF', 'BsmtRatio1', 'BsmtRatio2']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=_input1, x='BsmtRatio1', y='SalePrice', ax=ax[0])
sns.scatterplot(data=_input1, x='BsmtRatio2', y='SalePrice', ax=ax[1])
fig.show()
_input1['HasWoodDeck'] = _input1['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
_input0['HasWoodDeck'] = _input0['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
_input1['HasOpenPorch'] = _input1['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
_input0['HasOpenPorch'] = _input0['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
print(_input1[['WoodDeckSF', 'HasWoodDeck', 'OpenPorchSF', 'HasOpenPorch', 'SalePrice']].head())
wood_deck_median = _input1.loc[_input1['HasWoodDeck'] == 1, 'SalePrice'].median() / 1000
no_wood_deck_median = _input1[_input1['HasWoodDeck'] == 0]['SalePrice'].median() / 1000
(fig, ax) = plt.subplots(2, 2, figsize=(18, 8))
ax[0, 0].set_title('HasWoodDeck')
ax[0, 0].boxplot(x=[_input1[_input1['HasWoodDeck'] == 1]['SalePrice'], _input1[_input1['HasWoodDeck'] == 0]['SalePrice']], labels=[f'WoodDeck ({wood_deck_median})', f'No WoodDeck ({no_remode_median})'])
open_porch_median = _input1.loc[_input1['HasOpenPorch'] == 1, 'SalePrice'].median() / 1000
no_open_porch_median = _input1[_input1['HasOpenPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('HasOpenPorch')
ax[0, 1].boxplot(x=[_input1[_input1['HasOpenPorch'] == 1]['SalePrice'], _input1[_input1['HasOpenPorch'] == 0]['SalePrice']], labels=[f'OpenPorch ({open_porch_median})', f'No OpenPorch ({no_open_porch_median})'])
ax[1, 0].set_title('HasWoodDeck')
ax[1, 0].hist(_input1['HasWoodDeck'])
ax[1, 1].set_title('HasOpenPorch')
ax[1, 1].hist(_input1['HasOpenPorch'])
fig.show()
(fig, ax) = plt.subplots(4, 2, figsize=(18, 8))
fig.tight_layout()
_input1['HasSecondFloor'] = _input1['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
_input0['HasSecondFloor'] = _input0['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
second_floor_median = _input1.loc[_input1['HasSecondFloor'] == 1, 'SalePrice'].median() / 1000
no_second_floor_median = _input1[_input1['HasSecondFloor'] == 0]['SalePrice'].median() / 1000
ax[0, 0].set_title('SecondFloor')
ax[0, 0].boxplot(x=[_input1[_input1['HasSecondFloor'] == 1]['SalePrice'], _input1[_input1['HasSecondFloor'] == 0]['SalePrice']], labels=[f'SecondFloor ({second_floor_median})', f'No SecondFloor ({no_second_floor_median})'])
ax[1, 0].set_title('SecondFloor')
ax[1, 0].hist(_input1['HasSecondFloor'])
_input1['NoGrdBath'] = np.logical_and(_input1['FullBath'] == 0, _input1['HalfBath'] == 0).astype(int)
_input0['NoGrdBath'] = np.logical_and(_input0['FullBath'] == 0, _input0['HalfBath'] == 0).astype(int)
grd_bath_median = _input1.loc[_input1['NoGrdBath'] == 1, 'SalePrice'].median() / 1000
no_grd_bath_median = _input1[_input1['NoGrdBath'] == 0]['SalePrice'].median() / 1000
ax[2, 0].set_title('GrdBath')
ax[2, 0].boxplot(x=[_input1[_input1['NoGrdBath'] == 1]['SalePrice'], _input1[_input1['NoGrdBath'] == 0]['SalePrice']], labels=[f'GrdBath ({grd_bath_median})', f'No GrdBath ({no_grd_bath_median})'])
ax[3, 0].set_title('GrdBath')
ax[3, 0].hist(_input1['NoGrdBath'])
_input1['NoBsmtBath'] = pd.Series(np.logical_and(_input1['BsmtFullBath'] == 0, _input1['BsmtHalfBath'] == 0)).astype(int)
_input0['NoBsmtBath'] = pd.Series(np.logical_and(_input0['BsmtFullBath'] == 0, _input0['BsmtHalfBath'] == 0)).astype(int)
bsmt_bath_median = _input1.loc[_input1['NoBsmtBath'] == 1, 'SalePrice'].median() / 1000
no_bsmt_bath_median = _input1[_input1['NoBsmtBath'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('BsmtBath')
ax[0, 1].boxplot(x=[_input1[_input1['NoBsmtBath'] == 1]['SalePrice'], _input1[_input1['NoBsmtBath'] == 0]['SalePrice']], labels=[f'BsmtBath ({bsmt_bath_median})', f'No BsmtBath ({no_bsmt_bath_median})'])
ax[1, 1].set_title('BsmtBath')
ax[1, 1].hist(_input1['NoBsmtBath'])
_input1['NoBath'] = np.logical_and(_input1['NoBsmtBath'] == 1, _input1['NoGrdBath'] == 1).astype(int)
_input0['NoBath'] = np.logical_and(_input0['NoBsmtBath'] == 1, _input0['NoGrdBath'] == 1).astype(int)
bath_median = _input1.loc[_input1['NoBath'] == 1, 'SalePrice'].median() / 1000
no_bath_median = _input1[_input1['NoBath'] == 0]['SalePrice'].median() / 1000
ax[2, 1].set_title('Bath')
ax[2, 1].boxplot(x=[_input1[_input1['NoBath'] == 1]['SalePrice'], _input1[_input1['NoBath'] == 0]['SalePrice']], labels=[f'BsmtBath ({bath_median})', f'No BsmtBath ({no_bath_median})'])
ax[3, 1].set_title('Bath')
ax[3, 1].hist(_input1['NoBath'])
fig.show()
(fig, ax) = plt.subplots(4, 2, figsize=(18, 8))
fig.tight_layout()
_input1['HasEnclosedPorch'] = _input1['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
_input0['HasEnclosedPorch'] = _input0['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
enclosed_porch_median = _input1.loc[_input1['HasEnclosedPorch'] == 1, 'SalePrice'].median() / 1000
no_enclosed_porch_median = _input1[_input1['HasEnclosedPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 0].set_title('Enclosed Porch')
ax[0, 0].boxplot(x=[_input1[_input1['HasEnclosedPorch'] == 1]['SalePrice'], _input1[_input1['HasEnclosedPorch'] == 0]['SalePrice']], labels=[f'Enclosed Porch ({enclosed_porch_median})', f'No Enclosed Porch ({no_enclosed_porch_median})'])
ax[1, 0].set_title('Enclosed Porch')
ax[1, 0].hist(_input1['HasEnclosedPorch'])
_input1['Has3SsnPorch'] = _input1['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
_input0['Has3SsnPorch'] = _input0['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
three_ssn_median = _input1.loc[_input1['Has3SsnPorch'] == 1, 'SalePrice'].median() / 1000
no_three_ssn_median = _input1[_input1['Has3SsnPorch'] == 0]['SalePrice'].median() / 1000
ax[0, 1].set_title('3 Seasons Porch')
ax[0, 1].boxplot(x=[_input1[_input1['Has3SsnPorch'] == 1]['SalePrice'], _input1[_input1['Has3SsnPorch'] == 0]['SalePrice']], labels=[f'3 Seasons Porch ({three_ssn_median})', f'No 3 Seasons Porch ({no_three_ssn_median})'])
ax[1, 1].set_title('3 Seasons Porch')
ax[1, 1].hist(_input1['Has3SsnPorch'])
_input1['HasScreenPorch'] = _input1['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
_input0['HasScreenPorch'] = _input0['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
scrn_porch_median = _input1.loc[_input1['HasScreenPorch'] == 1, 'SalePrice'].median() / 1000
no_scrn_porch_median = _input1[_input1['HasScreenPorch'] == 0]['SalePrice'].median() / 1000
ax[2, 0].set_title('Screen Porch')
ax[2, 0].boxplot(x=[_input1[_input1['HasScreenPorch'] == 1]['SalePrice'], _input1[_input1['HasScreenPorch'] == 0]['SalePrice']], labels=[f'GrdBath ({scrn_porch_median})', f'No GrdBath ({no_scrn_porch_median})'])
ax[3, 0].set_title('Screen Porch')
ax[3, 0].hist(_input1['HasScreenPorch'])
_input1['HasPorch'] = np.logical_or(np.logical_or(_input1['HasOpenPorch'] == 1, _input1['HasEnclosedPorch'] == 1), np.logical_or(_input1['Has3SsnPorch'] == 1, _input1['HasScreenPorch'] == 1)).astype(int)
_input0['HasPorch'] = np.logical_or(np.logical_or(_input0['HasOpenPorch'] == 1, _input0['HasEnclosedPorch'] == 1), np.logical_or(_input0['Has3SsnPorch'] == 1, _input0['HasScreenPorch'] == 1)).astype(int)
porch_median = _input1.loc[_input1['HasPorch'] == 1, 'SalePrice'].median() / 1000
no_porch_median = _input1[_input1['HasPorch'] == 0]['SalePrice'].median() / 1000
ax[2, 1].set_title('Porch')
ax[2, 1].boxplot(x=[_input1[_input1['HasPorch'] == 1]['SalePrice'], _input1[_input1['HasPorch'] == 0]['SalePrice']], labels=[f'Porch ({porch_median})', f'No Porch ({no_porch_median})'])
ax[3, 1].set_title('Porch')
ax[3, 1].hist(_input1['HasPorch'])
print(_input1[['OpenPorchSF', 'HasOpenPorch', 'EnclosedPorch', 'HasEnclosedPorch', '3SsnPorch', 'Has3SsnPorch', 'ScreenPorch', 'HasScreenPorch', 'HasPorch', 'SalePrice']].head())
fig.show()
_input1['LotArea'].describe()
_input1['FrontageRatio'] = _input1['LotFrontage'] / _input1['LotArea']
_input0['FrontageRatio'] = _input0['LotFrontage'] / _input0['LotArea']
print(_input1[['LotFrontage', 'LotArea', 'FrontageRatio', 'SalePrice']].head())
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(data=_input1, x='BsmtRatio1', y='SalePrice', ax=ax[0])
ax[1].hist(_input1['FrontageRatio'])
fig.show()
print(_input1.YrSold.unique())
print(_input0.YrSold.unique())
_input1['YrSoldLog'] = np.log(_input1['YrSold'] - 2000)
_input1['YrSoldCube'] = _input1['YrSold'] ** 3
_input1['YrSoldSqrt'] = _input1['YrSold'] ** 0.5
_input0['YrSoldLog'] = np.log(_input0['YrSold'] - 2000)
_input0['YrSoldCube'] = _input0['YrSold'] ** 3
_input0['YrSoldSqrt'] = _input0['YrSold'] ** 0.5
print(_input1[['YrSold', 'YrSoldLog', 'YrSoldCube', 'YrSoldSqrt']].head())
(fig, ax) = plt.subplots(1, 4, figsize=(18, 4))
ax[0].set_title('YrSold')
ax[0].hist(_input1['YrSold'])
ax[1].set_title('YrSoldCube')
ax[1].hist(_input1['YrSoldCube'])
ax[2].set_title('YrSoldSqrt')
ax[2].hist(_input1['YrSoldSqrt'])
ax[3].set_title('YrSoldLog')
ax[3].hist(_input1['YrSoldLog'])
fig.show()
_input1['MSSubClassCbrt'] = _input1['MSSubClass'] ** (1 / 3)
_input1['MSSubClassSquare'] = _input1['MSSubClass'] ** 2
_input1['MSSubClassLog'] = np.log(_input1['MSSubClass'] + 1)
_input0['MSSubClassCbrt'] = _input0['MSSubClass'] ** (1 / 3)
_input0['MSSubClassSquare'] = _input0['MSSubClass'] ** 2
_input0['MSSubClassLog'] = np.log(_input0['MSSubClass'] + 1)
_input1['MoSoldSqrt'] = _input1['MoSold'] ** (1 / 2)
_input1['MoSoldCubic'] = _input1['MoSold'] ** 3
_input1['MoSoldLog'] = np.log(_input1['MoSold'] + 1)
_input0['MoSoldSqrt'] = _input0['MoSold'] ** (1 / 2)
_input0['MoSoldCubic'] = _input0['MoSold'] ** 3
_input0['MoSoldLog'] = np.log(_input0['MoSold'] + 1)
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('MSSubClass')
ax[0, 0].hist(_input1['MSSubClass'])
ax[0, 1].set_title('MSSubClassCbrt')
ax[0, 1].hist(_input1['MSSubClassCbrt'])
ax[0, 2].set_title('MSSubClassSquare')
ax[0, 2].hist(_input1['MSSubClassSquare'])
ax[0, 3].set_title('MSSubClassLog')
ax[0, 3].hist(_input1['MSSubClassLog'])
ax[1, 0].set_title('MoSold')
ax[1, 0].hist(_input1['MoSold'])
ax[1, 1].set_title('MoSoldSqrt')
ax[1, 1].hist(_input1['MoSoldSqrt'])
ax[1, 2].set_title('MoSoldCubic')
ax[1, 2].hist(_input1['MoSoldCubic'])
ax[1, 3].set_title('MoSoldLog')
ax[1, 3].hist(_input1['MoSoldLog'])
fig.show()
_input1['GarageYrBuiltCbrt'] = _input1['GarageYrBlt'] ** (1 / 3)
_input1['GarageYrBuiltSquare'] = _input1['GarageYrBlt'] ** 2
_input1['GarageYrBuiltLog'] = np.log(_input1['GarageYrBlt'] - 1800)
_input0['GarageYrBuiltCbrt'] = _input0['GarageYrBlt'] ** (1 / 3)
_input0['GarageYrBuiltSquare'] = _input0['GarageYrBlt'] ** 2
_input0['GarageYrBuiltLog'] = np.log(_input0['GarageYrBlt'] - 1800)
_input1['YearBuiltSqrt'] = _input1['YearBuilt'] ** (1 / 2)
_input1['YearBuiltCubic'] = _input1['YearBuilt'] ** 3
_input1['YearBuiltLog'] = np.log(_input1['YearBuilt'] - 1700)
_input0['YearBuiltSqrt'] = _input0['YearBuilt'] ** (1 / 2)
_input0['YearBuiltCubic'] = _input0['YearBuilt'] ** 3
_input0['YearBuiltLog'] = np.log(_input0['YearBuilt'] - 1700)
print(_input1[['YearBuilt', 'YearBuiltSqrt', 'YearBuiltCubic', 'YearBuiltLog', 'GarageYrBlt', 'GarageYrBuiltCbrt', 'GarageYrBuiltSquare', 'GarageYrBuiltLog']].head(500))
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('YearBuilt')
ax[0, 0].hist(_input1['YearBuilt'])
ax[0, 1].set_title('YearBuiltSqrt')
ax[0, 1].hist(_input1['YearBuiltSqrt'])
ax[0, 2].set_title('YearBuiltCubic')
ax[0, 2].hist(_input1['YearBuiltCubic'])
ax[0, 3].set_title('YearBuiltLog')
ax[0, 3].hist(_input1['YearBuiltLog'])
ax[1, 0].set_title('GarageYrBlt')
ax[1, 0].hist(_input1['GarageYrBlt'])
ax[1, 1].set_title('GarageYrBuiltCbrt')
ax[1, 1].hist(_input1['GarageYrBuiltCbrt'])
ax[1, 2].set_title('GarageYrBuiltSquare')
ax[1, 2].hist(_input1['GarageYrBuiltSquare'])
ax[1, 3].set_title('GarageYrBuiltLog')
ax[1, 3].hist(_input1['GarageYrBuiltLog'])
fig.show()
_input1['GarageAreaLog'] = np.log(_input1['GarageArea'] + 1)
_input0['GarageAreaLog'] = np.log(_input0['GarageArea'] + 1)
_input1['GarageAreaSquare'] = _input1['GarageArea'] ** 2
_input0['GarageAreaSquare'] = _input0['GarageArea'] ** 2
_input1['GarageAreaCbrt'] = _input1['GarageArea'] ** (1 / 3)
_input0['GarageAreaCbrt'] = _input0['GarageArea'] ** (1 / 3)
_input1['BsmtUnfSFLog'] = np.log(_input1['BsmtUnfSF'] + 5)
_input0['BsmtUnfSFLog'] = np.log(_input0['BsmtUnfSF'] + 5)
_input1['BsmtUnfSFCube'] = _input1['BsmtUnfSF'] ** 3
_input0['BsmtUnfSFCube'] = _input0['BsmtUnfSF'] ** 3
_input1['BsmtUnfSFSqrt'] = _input1['BsmtUnfSF'] ** (1 / 2)
_input0['BsmtUnfSFSqrt'] = _input0['BsmtUnfSF'] ** (1 / 2)
(fig, ax) = plt.subplots(2, 4, figsize=(18, 4))
fig.tight_layout()
ax[0, 0].set_title('GarageArea')
ax[0, 0].hist(_input1['GarageArea'])
ax[0, 1].set_title('GarageAreaLog')
ax[0, 1].hist(_input1['GarageAreaLog'])
ax[0, 2].set_title('GarageAreaSquare')
ax[0, 2].hist(_input1['GarageAreaSquare'])
ax[0, 3].set_title('GarageAreaCbrt')
ax[0, 3].hist(_input1['GarageAreaCbrt'])
ax[1, 0].set_title('BsmtUnfSF')
ax[1, 0].hist(_input1['BsmtUnfSF'])
ax[1, 1].set_title('BsmtUnfSFLog')
ax[1, 1].hist(_input1['BsmtUnfSFLog'])
ax[1, 2].set_title('GarageYrBuiltSquare')
ax[1, 2].hist(_input1['GarageYrBuiltSquare'])
ax[1, 3].set_title('BsmtUnfSFSqrt')
ax[1, 3].hist(_input1['BsmtUnfSFSqrt'])
fig.show()
_input1['OverallQualSquare'] = _input1['OverallQual'] ** 2
_input0['OverallQualSquare'] = _input0['OverallQual'] ** 2
_input1['OverallQualCube'] = _input1['OverallQual'] ** 3
_input0['OverallQualCube'] = _input0['OverallQual'] ** 3
_input1['OverallQualPlusCondSquare'] = _input1['OverallQualPlusCond'] ** 2
_input0['OverallQualPlusCondSquare'] = _input0['OverallQualPlusCond'] ** 2
_input1['OverallQualPlusCondCube'] = _input1['OverallQualPlusCond'] ** 3
_input0['OverallQualPlusCondCube'] = _input0['OverallQualPlusCond'] ** 3
(fig, ax) = plt.subplots(2, 3, figsize=(12, 4))
fig.tight_layout()
ax[0, 0].set_title('OverallQual')
ax[0, 0].hist(_input1['OverallQual'])
ax[0, 1].set_title('OverallQualSquare')
ax[0, 1].hist(_input1['OverallQualSquare'])
ax[0, 2].set_title('OverallQualCube')
ax[0, 2].hist(_input1['OverallQualCube'])
ax[1, 0].set_title('OverallQualPlusCond')
ax[1, 0].hist(_input1['OverallQualPlusCond'])
ax[1, 1].set_title('OverallQualPlusCondSquare')
ax[1, 1].hist(_input1['OverallQualPlusCondSquare'])
ax[1, 2].set_title('OverallQualPlusCondCube')
ax[1, 2].hist(_input1['OverallQualPlusCondCube'])
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
numeric_columns = _input0.select_dtypes('number').columns
_input1 = handle_multiple_outliers(_input1, numeric_columns)
_input0 = handle_multiple_outliers(_input0, numeric_columns)
for col in numeric_columns:
    plt.figure(figsize=(12, 1))
    ax = sns.boxplot(x=_input1[col])
_input1['MSSubClass'] = _input1['MSSubClass'].astype(str)
_input0['MSSubClass'] = _input0['MSSubClass'].astype(str)
_input1['MoSold'] = _input1['MoSold'].astype(str)
_input0['MoSold'] = _input0['MoSold'].astype(str)
_input1['YrSold'] = _input1['YrSold'].astype(str)
_input0['YrSold'] = _input0['YrSold'].astype(str)
_input1['OverallQualPlusCond'] = _input1['OverallQualPlusCond'].astype(str)
_input0['OverallQualPlusCond'] = _input0['OverallQualPlusCond'].astype(str)
_input1['OverallQual'] = _input1['OverallQual'].astype(str)
_input0['OverallQual'] = _input0['OverallQual'].astype(str)
_input1['OverallCond'] = _input1['OverallCond'].astype(str)
_input0['OverallCond'] = _input0['OverallCond'].astype(str)
print(_input1['OverallQual'].describe())
print(_input1['MoSold'].describe())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = _input0.select_dtypes('number').columns
X_num_train_scaled = scaler.fit_transform(_input1[numeric_columns])
print(X_num_train_scaled[:2])
from sklearn.linear_model import Lasso
X_lasso = X_num_train_scaled
y_lasso = _input1['SalePrice']
lasso = Lasso(tol=0.01, positive=True)