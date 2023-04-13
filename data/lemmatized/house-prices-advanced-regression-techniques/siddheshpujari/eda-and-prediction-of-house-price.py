import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
sns.set_style('darkgrid')
plt.style.use('fivethirtyeight')
import plotly.io as pio
pio.templates.default = 'gridon'
from plotly.subplots import make_subplots
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input1.shape
_input1.columns
_input1.describe().T
_input1.skew()
_input1.info()
_input1.columns
features_to_change = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'GarageCars']
int_to_object = ['MSSubClass', 'OverallQual', 'OverallCond', 'GarageCars']
for feature in int_to_object:
    _input1[feature] = _input1[feature].astype(object)
    _input0[feature] = _input0[feature].astype(object)
_input1 = _input1.drop(columns=['Id'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['Id'], axis=1, inplace=False)
_input1.isna().sum().sum()
_input0.isna().sum().sum()
_input1.isna().sum()
temp = list(_input1.isna().sum())
lst = []
i = 0
for col in _input1.columns:
    insert_lst = [col, temp[i]]
    lst.append(insert_lst)
    i += 1
temp_df = pd.DataFrame(data=lst, columns=['Column_Name', 'Missing_Values'])
fig = px.bar(temp_df.sort_values(by='Missing_Values'), x='Missing_Values', y='Column_Name', orientation='h', height=1500, width=900, color='Missing_Values', text='Missing_Values', title='Missing values in train dataset')
fig.update_traces(textposition='outside')
fig.show()
temp_df[temp_df['Missing_Values'] > 0].sort_values(by='Missing_Values', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
temp = list(_input0.isna().sum())
lst = []
i = 0
for col in _input0.columns:
    insert_lst = [col, temp[i]]
    lst.append(insert_lst)
    i += 1
temp_df = pd.DataFrame(data=lst, columns=['Column_Name', 'Missing_Values'])
temp_df[temp_df['Missing_Values'] > 0].sort_values(by='Missing_Values', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
temp_df[temp_df['Missing_Values'] > 0].sort_values(by='Missing_Values', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
plt.figure(figsize=(15, 5))
features_to_examine = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
temp = _input1[features_to_examine]
colors = ['', 'red', 'blue', 'green']
i = 1
for col in temp.columns:
    plt.subplot(1, 3, i)
    a1 = sns.boxplot(data=temp, y=col, color=colors[i])
    i += 1
(_input1['LotFrontage'].mean(), _input1['LotFrontage'].median())
(_input1['MasVnrArea'].mean(), _input1['MasVnrArea'].median())
(_input1['GarageYrBlt'].mean(), _input1['GarageYrBlt'].median())
features_to_examine
for col in features_to_examine:
    _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
for col in features_to_examine:
    _input0[col] = _input0[col].fillna(_input0[col].median(), inplace=False)
_input1[features_to_examine].isna().sum()
_input0[features_to_examine].isna().sum()
features_to_examine = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
_input1['PoolQC'].value_counts()
_input1['MiscFeature'].value_counts()
_input1['Alley'].value_counts()
_input1['Fence'].value_counts()
_input1 = _input1.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=False)
_input1['FireplaceQu'].value_counts()
_input1[_input1['FireplaceQu'].isnull()][['Fireplaces', 'FireplaceQu']]
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('NotAvailable')
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('NotAvailable')
features_to_examine = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
_input1['MasVnrType'].isna().sum()
_input1['MasVnrType'].unique()
_input1[_input1['MasVnrType'].isnull()][['MasVnrType', 'MasVnrArea']]
_input1['MasVnrType'].mode()
_input1[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isna().sum()
_input1[_input1['BsmtQual'].isnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
_input1[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].mode()
_input1['Electrical'].isna().sum()
_input1['Electrical'].mode()
_input1[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].isna().sum()
_input1[_input1['GarageType'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
_input1[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].mode()
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
fill_with_No_Bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
fill_with_No_Grg = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in fill_with_No_Bsmt:
    _input1[col] = _input1[col].fillna('No_Bsmt', inplace=False)
    _input0[col] = _input0[col].fillna('No_Bsmt', inplace=False)
for col in fill_with_No_Grg:
    _input1[col] = _input1[col].fillna('No_Grg', inplace=False)
    _input0[col] = _input0[col].fillna('No_Grg', inplace=False)
_input1['Electrical'] = _input1['Electrical'].fillna('SBrkr', inplace=False)
_input0['Electrical'] = _input0['Electrical'].fillna('SBrkr', inplace=False)
_input1.isna().sum().sum()
_input0.isna().sum().sum()
temp = list(_input0.isna().sum())
lst = []
i = 0
for col in _input0.columns:
    insert_lst = [col, temp[i]]
    lst.append(insert_lst)
    i += 1
temp_df = pd.DataFrame(data=lst, columns=['Column_Name', 'Missing_Values'])
temp_df = temp_df[temp_df['Missing_Values'] > 0].sort_values(by='Missing_Values', ascending=False).reset_index(drop=True)
temp_df.style.background_gradient(cmap='Reds')
features_to_examine = temp_df['Column_Name'].unique().tolist()
features_to_examine
_input0[features_to_examine].info()
cat = [col for col in features_to_examine if _input0[col].dtype == 'O']
for feature in cat:
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0])
num = [col for col in features_to_examine if _input0[col].dtype != 'O']
for feature in num:
    _input0[feature] = _input0[feature].fillna(_input0[feature].median())
_input0.isna().sum().sum()
_input1['SalePrice'].describe()
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Histogram(x=_input1['SalePrice']), row=1, col=1)
fig.add_trace(go.Box(y=_input1['SalePrice'], boxpoints='all', line_color='orange'), row=1, col=2)
fig.update_layout(height=500, showlegend=False, title_text='Sale Price Distribution and Box Plot')
discrete = []
for col in _input1.columns:
    if _input1[col].dtype == 'int64' and len(_input1[col].unique()) <= 15:
        discrete.append(col)
print('Discrete Features :: \n\n{}'.format(discrete))
for col in discrete:
    print('{} has {} unique values.'.format(col, _input1[col].unique()))
from numpy import median
fig = plt.figure(constrained_layout=True, figsize=(15, 25))
gs = GridSpec(6, 3, figure=fig)
plt.subplot(gs[0, :])
a1 = sns.barplot(data=_input1, x='TotRmsAbvGrd', y='SalePrice', estimator=median, palette='hot')
plt.xlabel('TotRmsAbvGrd', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[1, :-1])
a1 = sns.barplot(data=_input1, x='BedroomAbvGr', y='SalePrice', estimator=median, palette='magma')
plt.xlabel('BedroomAbvGrd', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[1, -1])
a1 = sns.barplot(data=_input1, x='KitchenAbvGr', y='SalePrice', estimator=median, palette='Purples_r')
plt.xlabel('KitchenAbvGr', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[2, :-1])
a1 = sns.barplot(data=_input1, x='BsmtFullBath', y='SalePrice', estimator=median, palette='magma')
plt.xlabel('BsmtFullBath', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[2, -1])
a1 = sns.barplot(data=_input1, x='FullBath', y='SalePrice', estimator=median, palette='Purples_r')
plt.xlabel('FullBath', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[3, :-1])
a1 = sns.barplot(data=_input1, x='BsmtHalfBath', y='SalePrice', estimator=median, palette='magma')
plt.xlabel('BsmtHalfBath', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[3, -1])
a1 = sns.barplot(data=_input1, x='HalfBath', y='SalePrice', estimator=median, palette='Purples_r')
plt.xlabel('HalfBath', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[4, :-2])
a1 = sns.barplot(data=_input1, x='Fireplaces', y='SalePrice', estimator=median)
plt.xlabel('Fireplaces', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[4, -2:])
a1 = sns.barplot(data=_input1, x='PoolArea', y='SalePrice', estimator=median, palette='icefire')
plt.xlabel('PoolArea', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[5, :-2])
a1 = sns.barplot(data=_input1, x='YrSold', y='SalePrice', estimator=median)
plt.xlabel('YrSold', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.subplot(gs[5, -2:])
a1 = sns.barplot(data=_input1, x='MoSold', y='SalePrice', estimator=median, palette='icefire')
plt.xlabel('MoSold', fontsize=15)
plt.ylabel('Average SalePrice', fontsize=15)
plt.suptitle('Discrete Numerical Analysis', fontsize=20)
num = []
for col in _input1.columns:
    if _input1[col].dtype == 'int64' and col not in ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'GarageYrBlt'] and (col not in discrete):
        num.append(col)
print('Numerical Features :: \n\n{}'.format(num))
df_corr = _input1[num].iloc[:, 0:10]
df_corr['SalePrice'] = _input1['SalePrice']
corr = df_corr.corr()
fig = plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, linewidths=0.5, cmap='coolwarm', vmin=-1, vmax=1, center=0)
df_corr = _input1[num].iloc[:, 10:]
df_corr['SalePrice'] = _input1['SalePrice']
corr = df_corr.corr()
fig = plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, linewidths=0.5, cmap='coolwarm', vmin=-1, vmax=1, center=0)
_input1.columns
fig = make_subplots(rows=2, cols=2)
features_to_examine = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']
i = 0
for row in range(1, 3):
    for col in range(1, 3):
        fig.add_trace(go.Scatter(y=_input1['SalePrice'], x=_input1[features_to_examine[i]], name=features_to_examine[i], mode='markers'), row=row, col=col)
        i += 1
fig.update_layout(height=1000, showlegend=True, title_text='Positive Correlated features with Sale Price')
fig.show()
cat = []
for col in _input1.columns:
    if _input1[col].dtype == 'object':
        cat.append(col)
print('Categorical variables :: \n\n{}'.format(cat))
cat1 = ['MSSubClass']
df_new = _input1.copy()
df_new['MSSubClass'] = df_new['MSSubClass'].replace({20: '1_STORY_NEWER', 30: '1_STORY_OLDER', 40: '1_STORY_ATTIC', 45: '1_1/2_UNFINISHED', 50: '1_1/2_FINISHED', 60: '2_STORY_NEWER', 70: '2_STORY_OLDER', 75: '2_1/2_STOPY', 80: 'SPLIT/MULTILEVEL', 85: 'SPLIT_FOYER', 90: 'DUPLEX', 120: '1_STORY_PUD', 150: '1_1/2_STORY_PUD', 160: '2_STORY_PUD', 180: 'PUD_MULTILEVEL', 190: '2_FAMILY_CONVERSION'})
df_new['MSSubClass'] = df_new['MSSubClass'].astype(object)
fig = plt.figure(figsize=(15, 5))
table = df_new.groupby(['MSSubClass'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
ax = sns.boxplot(data=df_new, x='MSSubClass', y='SalePrice', order=table['MSSubClass'].to_list(), palette='Set1')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
fig.suptitle('Category 1 : Type Of Dwellings', fontsize=25)
table = df_new.groupby(['MSSubClass'])['MSSubClass', 'SalePrice'].agg({'MSSubClass': 'count', 'SalePrice': 'median'})
table = table.sort_values(by='SalePrice', ascending=False)
table.style.background_gradient(cmap='Reds')
cat2 = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope']
fig = plt.figure(constrained_layout=True, figsize=(15, 20))
gs = gridspec.GridSpec(4, 2, figure=fig)
plt.subplot(gs[0, 0])
table = _input1.groupby(['MSZoning'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='MSZoning', y='SalePrice', order=table['MSZoning'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[0, 1])
table = _input1.groupby(['LandContour'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='LandContour', y='SalePrice', order=table['LandContour'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[1, :])
a1 = sns.distplot(_input1[_input1['Street'] == 'Grvl']['SalePrice'])
a1 = sns.distplot(_input1[_input1['Street'] == 'Pave']['SalePrice'])
plt.legend('upper right', labels=['Grvl', 'Pave'])
plt.xlabel('SalePrice')
plt.title('Street')
plt.subplot(gs[2, 0])
table = _input1.groupby(['LotShape'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='LotShape', y='SalePrice', order=table['LotShape'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[2, 1])
table = _input1.groupby(['LotConfig'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='LotConfig', y='SalePrice', order=table['LotConfig'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[3, :])
a1 = sns.distplot(_input1[_input1['LandSlope'] == 'Gtl']['SalePrice'])
a1 = sns.distplot(_input1[_input1['LandSlope'] == 'Mod']['SalePrice'])
a1 = sns.distplot(_input1[_input1['LandSlope'] == 'Sev']['SalePrice'])
plt.legend('upper right', labels=['Gtl', 'Mod', 'Sev'])
plt.xlabel('SalePrice')
plt.title('Land Slope')
fig.suptitle('Category 2 : Structure of Land and Property', fontsize=25)
cat2 = ['OverallQual', 'OverallCond']
df_new['OverallQual'] = df_new['OverallQual'].replace({10: 'Very Exc', 9: 'Exc', 8: 'VG', 7: 'Good', 6: 'Abv Avg', 5: 'Avg', 4: 'Bel Avg', 3: 'Fair', 2: 'Poor', 1: 'Very Poor'})
df_new['OverallQual'] = df_new['OverallQual'].astype(object)
df_new['OverallCond'] = df_new['OverallCond'].replace({10: 'Very Exc', 9: 'Exc', 8: 'VG', 7: 'Good', 6: 'Abv Avg', 5: 'Avg', 4: 'Bel Avg', 3: 'Fair', 2: 'Poor', 1: 'Very Poor'})
df_new['OverallCond'] = df_new['OverallCond'].astype(object)
fig = plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 1)
table = df_new.groupby(['OverallQual'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=df_new, x='OverallQual', y='SalePrice', order=table['OverallQual'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(2, 1, 2)
table = df_new.groupby(['OverallCond'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=df_new, x='OverallCond', y='SalePrice', order=table['OverallCond'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
fig.suptitle('Category 3 : Overall Quality and Condition', fontsize=25)
fig = px.histogram(df_new, x='SalePrice', color='OverallQual', barmode='overlay', title='Overall Quality of the house')
fig.update_layout(height=500)
fig.show()
fig = px.histogram(df_new, x='SalePrice', color='OverallCond', barmode='overlay', title='Overall Condition of the house')
fig.update_layout(height=500)
fig.show()
cat2 = ['Neighborhood', 'Condition1', 'Condition2', 'Utilities', 'BldgType', 'HouseStyle']
fig = plt.figure(constrained_layout=True, figsize=(15, 18))
gs = GridSpec(4, 3, figure=fig)
plt.subplot(gs[0, :])
table = _input1.groupby(['Neighborhood'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='Neighborhood', y='SalePrice', order=table['Neighborhood'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[1, :-1])
table = _input1.groupby(['Condition1'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='Condition1', y='SalePrice', order=table['Condition1'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[1, -1])
table = _input1.groupby(['Utilities'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.stripplot(data=_input1, x='Utilities', y='SalePrice', order=table['Utilities'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[2, :-1])
table = _input1.groupby(['Condition2'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='Condition2', y='SalePrice', order=table['Condition2'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[2, -1])
table = _input1.groupby(['BldgType'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.stripplot(data=_input1, x='BldgType', y='SalePrice', order=table['BldgType'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[3, :])
table = _input1.groupby(['HouseStyle'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='HouseStyle', y='SalePrice', order=table['HouseStyle'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
fig.suptitle('Location and Style', fontsize=25)
table = df_new.groupby(['Neighborhood'], as_index=False)['SalePrice'].median()
table = table.sort_values(by='SalePrice', ascending=False)
table.style.background_gradient(cmap='Reds')
cat3 = ['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation']
fig = plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 2, 1)
table = _input1.groupby(['Foundation'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='Foundation', y='SalePrice', order=table['Foundation'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(2, 2, 2)
table = _input1.groupby(['RoofMatl'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='RoofMatl', y='SalePrice', order=table['RoofMatl'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(2, 2, 3)
table = _input1.groupby(['Exterior1st'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='Exterior1st', y='SalePrice', order=table['Exterior1st'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(2, 2, 4)
table = _input1.groupby(['Exterior2nd'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.stripplot(data=_input1, x='Exterior2nd', y='SalePrice', order=table['Exterior2nd'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
fig = px.histogram(_input1, x='SalePrice', color='RoofStyle', barmode='overlay', title='RoofStyle')
fig.update_layout(height=400)
fig.show()
fig = px.histogram(_input1, x='SalePrice', color='MasVnrType', barmode='overlay', title='Mason Veneer Type')
fig.update_layout(height=400)
fig.show()
table = df_new.groupby(['RoofStyle'], as_index=False)['SalePrice'].median()
table = table.sort_values(by='SalePrice', ascending=False)
table.style.background_gradient(cmap='Reds')
table = df_new.groupby(['MasVnrType'], as_index=False)['SalePrice'].median()
table = table.sort_values(by='SalePrice', ascending=False)
table.style.background_gradient(cmap='Greys')
cat6 = ['ExterQual', 'ExterCond']
fig = px.histogram(_input1, x='SalePrice', color='ExterQual', barmode='overlay', title='Exterior Quality')
fig.update_layout(height=400)
fig.show()
fig = px.histogram(_input1, x='SalePrice', color='ExterCond', barmode='overlay', title='Exterior Condition')
fig.update_layout(height=400)
fig.show()
cat7 = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Box(y=_input1['SalePrice'], x=_input1['BsmtQual'], name='Basement Quality'), row=1, col=1)
fig.add_trace(go.Box(y=_input1['SalePrice'], x=_input1['BsmtExposure'], name='Basement Exposure'), row=1, col=2)
fig.update_layout(title_text='Basement Quality and Exposure', height=400)
fig.show()
fig = px.histogram(_input1, x='SalePrice', color='BsmtCond', barmode='overlay', height=400, title='Basement Condition')
fig.show()
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Violin(y=_input1['SalePrice'], x=_input1['BsmtFinType1'], name='Basement_Finish Type1 '), row=1, col=1)
fig.add_trace(go.Violin(y=_input1['SalePrice'], x=_input1['BsmtFinType2'], name='Basement_Finish Type2'), row=1, col=2)
fig.update_layout(title_text='Basement Finish Type 1 and 2', height=400)
fig.show()
cat8 = ['Heating', 'HeatingQC', 'CentralAir', 'Electrical']
import matplotlib.gridspec as gridspec
fig = plt.figure(tight_layout=True, figsize=(15, 12))
gs = gridspec.GridSpec(2, 2)
plt.subplot(gs[0, 0])
ax1 = sns.boxplot(data=_input1, x='Heating', y='SalePrice')
plt.subplot(gs[0, 1])
ax1 = sns.boxplot(data=_input1, x='HeatingQC', y='SalePrice')
plt.subplot(gs[1, :])
sns.distplot(_input1[_input1['CentralAir'] == 'Y']['SalePrice'])
sns.distplot(_input1[_input1['CentralAir'] == 'N']['SalePrice'])
plt.legend('upper right', labels=['Yes', 'No'])
plt.xlabel('SalePrice')
plt.title('Central Air Conditioning')
px.histogram(_input1, x='SalePrice', color='Electrical', barmode='overlay', title='Electrical System')
cat9 = ['KitchenQual', 'Functional', 'FireplaceQu']
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Violin(y=_input1['SalePrice'], x=_input1['FireplaceQu'], name='Fireplace Quality'), row=1, col=1)
fig.add_trace(go.Box(y=_input1['SalePrice'], x=_input1['KitchenQual'], name='Kitchen Quality'), row=1, col=2)
fig.update_layout(showlegend=True, title_text='Fireplace and Kitchen Quality')
fig.show()
fig = px.histogram(_input1, x='SalePrice', color='Functional', barmode='overlay', title='Functional')
fig.show()
cat8 = ['GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond']
fig = plt.figure(constrained_layout=True, figsize=(15, 15))
gs = GridSpec(3, 3, figure=fig)
plt.subplot(gs[0, :])
a1 = sns.boxenplot(data=_input1, x='GarageType', y='SalePrice')
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[1, :-1])
sns.distplot(_input1[_input1['GarageFinish'] == 'RFn']['SalePrice'])
sns.distplot(_input1[_input1['GarageFinish'] == 'Unf']['SalePrice'])
sns.distplot(_input1[_input1['GarageFinish'] == 'Fin']['SalePrice'])
sns.distplot(_input1[_input1['GarageFinish'] == 'Not Known']['SalePrice'])
plt.legend('upper right', labels=['RFn', 'Unf', 'Fin', 'Not Known'])
plt.xlabel('SalePrice')
plt.title('Garage Finish')
plt.subplot(gs[1:, -1])
a1 = sns.boxplot(data=_input1, x='GarageCars', y='SalePrice')
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[-1, 0])
a1 = sns.stripplot(data=_input1, x='GarageQual', y='SalePrice')
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(gs[-1, -2])
a1 = sns.stripplot(data=_input1, x='GarageCond', y='SalePrice')
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
cat9 = ['PavedDrive']
px.histogram(_input1, x='SalePrice', color='PavedDrive', barmode='overlay', title='Paved Driveway')
cat10 = ['SaleType', 'SaleCondition']
fig = plt.figure(figsize=(15, 15))
plt.subplot(2, 1, 1)
table = _input1.groupby(['SaleType'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxplot(data=_input1, x='SaleType', y='SalePrice', order=table['SaleType'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
plt.subplot(2, 1, 2)
table = _input1.groupby(['SaleCondition'], as_index=False)['SalePrice'].median().sort_values(by='SalePrice', ascending=False)
a1 = sns.boxenplot(data=_input1, x='SaleCondition', y='SalePrice', order=table['SaleCondition'].to_list())
a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
fig.suptitle('Sale Type and Condition', fontsize=25)
year_features = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']
year_features = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']
for feature in year_features:
    ax = _input1.groupby([feature])['SalePrice'].median().plot()
    plt.ylabel('Median House Price')
data = _input1.copy()
data['Sold-Built'] = data['YrSold'] - data['YearBuilt']
fig = px.scatter(data, x='Sold-Built', y='SalePrice', width=700)
fig.show()
data = _input1.copy()
data['Sold-Remodelled'] = data['YrSold'] - data['YearRemodAdd']
fig = px.scatter(data, x='Sold-Remodelled', y='SalePrice', width=700)
fig.show()
data = _input1.copy()
data['Sold-GarageBuilt'] = data['YrSold'] - data['GarageYrBlt']
fig = px.scatter(data, x='Sold-GarageBuilt', y='SalePrice', width=700)
fig.show()
import copy
dataset = _input1.copy()
object_to_int = ['MSSubClass', 'OverallQual', 'OverallCond', 'GarageCars']
for feature in object_to_int:
    dataset[feature] = dataset[feature].astype(int)
_input0['MSSubClass'] = _input0['MSSubClass'].astype(int)
_input0['OverallQual'] = _input0['OverallQual'].astype(int)
_input0['OverallCond'] = _input0['OverallCond'].astype(int)
datetime = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
continuous = []
for col in dataset.columns:
    if dataset[col].dtype != 'O' and len(dataset[col].unique()) > 16 and (col not in datetime):
        continuous.append(col)
skewed_features = []
for col in continuous:
    if dataset[col].skew() > 0 or dataset[col].skew() < 0:
        skewed_features.append(col)
apply_log = []
for col in skewed_features:
    if 0 not in dataset[col].unique():
        apply_log.append(col)
print('Before Log Transformation........................')
for col in apply_log:
    print('{} --- {}'.format(col, dataset[col].skew()))
for feature in apply_log:
    dataset[feature] = np.log(dataset[feature])
print('\nAfter Log Transformation.........................')
for col in apply_log:
    print('{} --- {}'.format(col, dataset[col].skew()))
for feature in apply_log:
    fig = sns.distplot(dataset[feature])
categorical = []
for col in dataset.columns:
    if dataset[col].dtype == 'O':
        categorical.append(col)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encoders = {}
for column in categorical:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encoders = {}
for column in categorical:
    label_encoders[column] = LabelEncoder()
    _input0[column] = label_encoders[column].fit_transform(_input0[column])
"for feature in categorical:\n    #for train\n    labels_ordered=dataset.groupby([feature])['SalePrice'].median().sort_values().index\n    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}\n    dataset[feature]=dataset[feature].map(labels_ordered)\n    \n    #same for test\n    labels_ordered=test.groupby([feature])['SalePrice'].median().sort_values().index\n    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}\n    test[feature]=test[feature].map(labels_ordered)"
dataset[categorical]
_input0[categorical]
y = dataset['SalePrice']
X = dataset.drop(columns=['SalePrice'], axis=1)
columns_x = X.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=[columns_x])
X.head()
columns_test = _input0.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
_input0 = scaler.fit_transform(_input0)
_input0 = pd.DataFrame(_input0, columns=[columns_test])
_input0.head()
from sklearn.model_selection import train_test_split as tts
(X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))