import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import os
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train
train.describe()
train['logSalePrice'] = np.log(train.SalePrice)
fig = px.histogram(train, x='SalePrice', title='Distribution of SalePrice', height=400)
fig.show()
fig1 = px.violin(train, x='SalePrice', title='Violin Plot for SalePrice', height=300)
fig1.update_traces(box_visible=True, meanline_visible=True)
fig1.show()
fig2 = px.violin(train, x='logSalePrice', title='Violin Plot for Log(SalePrice)', height=300)
fig2.update_traces(box_visible=True, meanline_visible=True)
fig2.show()
k = train.isnull().sum()
k[k > 0].sort_values(ascending=False)
numcols = train._get_numeric_data().columns
train_num = train.loc[:, numcols]
cols = train_num.columns
tup = tuple(cols[1:])
cols = train_num.columns
fig = make_subplots(rows=9, cols=4, shared_yaxes=True, subplot_titles=tup)
k = 1
j = 1
for i in range(1, 36):
    fig.add_trace(go.Scatter(y=train_num['SalePrice'], x=train_num.iloc[:, i], mode='markers', name=cols[i]), row=k, col=j)
    j = j % 4
    j = j + 1
    if i % 4 == 0:
        k = k + 1
fig.update_layout(height=1800, width=800, title_text='Dependency between SalePrice & Continous Variables', showlegend=False)
fig.show()
train_str_data = train.select_dtypes(include='object')
train_str_data['SalePrice'] = train['SalePrice']
cols = train_str_data.columns
tup = tuple(cols[1:])
fig = make_subplots(rows=10, cols=4, shared_yaxes=True, subplot_titles=tup)
row = 1
col = 1
for i in range(1, 41):
    uniqvals = train_str_data.iloc[:, i].unique()
    lenvals = len(uniqvals)
    for j in range(lenvals):
        k = train_str_data.iloc[:, i] == uniqvals[j]
        df = train_str_data.loc[k]
        fig.add_trace(go.Violin(x=df.iloc[:, i], y=df['SalePrice']), row=row, col=col)
    col = col % 4
    col = col + 1
    if i % 4 == 0:
        row = row + 1
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(height=2800, width=800, title_text='Dependency between SalePrice & Categorical Variables', showlegend=False)
fig.show()
fig = px.violin(train, y='SalePrice', x='Neighborhood', box=True, color='Neighborhood', title='Distribution of SalePrice by Neighborhood')
fig.update_layout(showlegend=False)
fig.show()
neighborhood_md = train.groupby('Neighborhood').agg({'SalePrice': 'median'}).reset_index().sort_values(by='SalePrice', ascending=False)
neighborhood_md
fig1 = px.bar(neighborhood_md, y='SalePrice', x='Neighborhood', color='SalePrice', title='Median SalePrice by Neighborhood')
fig1.update_layout(showlegend=False)
fig1.show()
condition = train.groupby(['Condition1', 'Condition2']).agg({'SalePrice': 'median', 'Utilities': 'count'}).reset_index().sort_values(by='SalePrice', ascending=False)
trans = condition.pivot(index='Condition1', columns='Condition2', values='SalePrice')
trans = trans.fillna(0)
cm = sns.light_palette('green', as_cmap=True)
s = trans.style.background_gradient(cmap=cm)
s
condition = condition.rename(columns={'SalePrice': 'avg_sale_price_cond', 'Utilities': 'total_records'})
condition.sort_values(by='total_records', ascending=False, inplace=True)
condition
gp = train.groupby(['YrSold', 'MoSold']).agg({'SalePrice': 'median', 'LotArea': 'count'}).reset_index()
gp['MoSold'] = gp['MoSold'].apply(str)
gp['YrSold'] = gp['YrSold'].apply(str)
gp['month_year'] = gp['YrSold'] + '-' + gp['MoSold']
gp.columns = ['YrSold', 'MoSold', 'SalePrice', 'SaleCount', 'month_year']
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(go.Bar(x=gp['month_year'], y=gp['SalePrice'], name='SalePrice'), secondary_y=False)
fig.add_trace(go.Scatter(x=gp['month_year'], y=gp['SaleCount'], name='SaleCount'), secondary_y=True)
fig.update_layout(title_text='Median SalePrice and SaleCount over Time')
fig.show()
neighborhood_md.loc[neighborhood_md['SalePrice'] >= 5000, 'neighborhood_flag'] = 0
neighborhood_md.loc[neighborhood_md['SalePrice'] >= 150000, 'neighborhood_flag'] = 1
neighborhood_md.loc[neighborhood_md['SalePrice'] >= 200000, 'neighborhood_flag'] = 2
neighborhood_md.loc[neighborhood_md['SalePrice'] >= 250000, 'neighborhood_flag'] = 3
del neighborhood_md['SalePrice']
neighborhood_md.columns = ['Neighborhood', 'neighborhood_flag']
train = pd.merge(train, neighborhood_md, left_on='Neighborhood', right_on='Neighborhood')
test = pd.merge(test, neighborhood_md, left_on='Neighborhood', right_on='Neighborhood')
train['tot_bath'] = train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath'] + train['FullBath'] + 0.5 * train['HalfBath']
train['bsmt_bath'] = train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath']
train['bed_bath_kitch'] = train['tot_bath'] + train['BedroomAbvGr'] + train['KitchenAbvGr']
train['area_floors'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['BsmtFinSF1'] + train['BsmtFinSF2']
train['bsmt_by_total'] = (train['BsmtFinSF1'] + train['BsmtFinSF2']) / train['area_floors']
train['unf_bsmt'] = (train['BsmtUnfSF'] / train['TotalBsmtSF']).fillna(0)
train['unf_bsmt'].replace([np.inf, -np.inf], 0)
train['porch_area_tot'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
train['wood_deck_porch'] = (train['WoodDeckSF'] / train['porch_area_tot']).replace(np.inf, 0)
train['sale_built_yr'] = train['YrSold'] - train['YearBuilt']
train['remod_built_yr'] = train['YearRemodAdd'] - train['YearBuilt']
train['new_flag'] = 0
train.loc[train['YrSold'] - train['YearBuilt'], 'new_flag'] = 1
train['remod_flag'] = 0
train.loc[train['remod_built_yr'] >= 2, 'remod_flag'] = 1
train['floor_by_lot'] = train['area_floors'] / train['LotArea']
train['garage_area_per_car'] = (train['GarageArea'] / train['GarageCars']).fillna(0)
train['garage_area_per_car'] = train['garage_area_per_car'].replace([np.inf, -np.inf], 0)
test['tot_bath'] = test['BsmtFullBath'] + 0.5 * test['BsmtHalfBath'] + test['FullBath'] + 0.5 * test['HalfBath']
test['bsmt_bath'] = test['BsmtFullBath'] + 0.5 * test['BsmtHalfBath']
test['bed_bath_kitch'] = test['tot_bath'] + test['BedroomAbvGr'] + test['KitchenAbvGr']
test['area_floors'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['BsmtFinSF1'] + test['BsmtFinSF2']
test['bsmt_by_total'] = (test['BsmtFinSF1'] + test['BsmtFinSF2']) / test['area_floors']
test['unf_bsmt'] = (test['BsmtUnfSF'] / test['TotalBsmtSF']).fillna(0)
test['unf_bsmt'].replace([np.inf, -np.inf], 0)
test['porch_area_tot'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
test['wood_deck_porch'] = (test['WoodDeckSF'] / test['porch_area_tot']).replace(np.inf, 0)
test['sale_built_yr'] = test['YrSold'] - test['YearBuilt']
test['remod_built_yr'] = test['YearRemodAdd'] - test['YearBuilt']
test['new_flag'] = 0
test.loc[test['YrSold'] - test['YearBuilt'] == 0, 'new_flag'] = 1
test['remod_flag'] = 0
test.loc[test['remod_built_yr'] >= 2, 'remod_flag'] = 1
test['floor_by_lot'] = test['area_floors'] / test['LotArea']
test['garage_area_per_car'] = (test['GarageArea'] / test['GarageCars']).fillna(0)
test['garage_area_per_car'] = test['garage_area_per_car'].replace([np.inf, -np.inf], 0)
mapping = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
mapping_shape = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
mapping_contour = {'Bnk': 0, 'Lvl': 1, 'Low': 2, 'HLS': 3}
mapping_ms_zoning = {'FV': 4, 'RL': 3, 'RH': 2, 'RM': 1, 'C (all)': 0, 'NA': -1}
mapping_paved_Drive = {'Y': 2, 'P': 1, 'N': 0}
mapping_utilities = {'AllPub': 1, 'NoSeWa': 0, 'NoSeWr': 0, 'ELO': 0, 'NA': 0}
mapping_functional = {'Typ': 4, 'Min1': 3, 'Min2': 3, 'Mod': 2, 'Maj1': 1, 'Maj2': 1, 'Sev': 0, 'Sal': 0, 'NA': 0}
l_col = ['BsmtCond', 'BsmtQual', 'ExterQual', 'ExterCond', 'HeatingQC', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual']
l_col_zon = ['MSZoning']
l_col_pav = ['PavedDrive']
l_col_shape = ['LotShape']
l_col_util = ['Utilities']
l_col_con = ['LandContour']
l_col_fun = ['Functional']

def mapping_var(mapping, df, varlist):
    for i in range(len(varlist)):
        varname = varlist[i]
        varname_o = varname + '_o'
        df[varname] = df[varname].fillna('NA')
        df[varname_o] = df[varname].apply(lambda x: mapping[x])
    return df
train = mapping_var(mapping, train, l_col)
train = mapping_var(mapping_ms_zoning, train, l_col_zon)
train = mapping_var(mapping_paved_Drive, train, l_col_pav)
train = mapping_var(mapping_shape, train, l_col_shape)
train = mapping_var(mapping_utilities, train, l_col_util)
train = mapping_var(mapping_contour, train, l_col_con)
train = mapping_var(mapping_functional, train, l_col_fun)
test = mapping_var(mapping, test, l_col)
test = mapping_var(mapping_ms_zoning, test, l_col_zon)
test = mapping_var(mapping_paved_Drive, test, l_col_pav)
test = mapping_var(mapping_shape, test, l_col_shape)
test = mapping_var(mapping_utilities, test, l_col_util)
test = mapping_var(mapping_contour, test, l_col_con)
test = mapping_var(mapping_functional, test, l_col_fun)
labelencoder = LabelEncoder()
train.loc[:, 'Street'] = labelencoder.fit_transform(train.loc[:, 'Street'])
train.loc[:, 'CentralAir'] = labelencoder.fit_transform(train.loc[:, 'CentralAir'])
test.loc[:, 'Street'] = labelencoder.fit_transform(test.loc[:, 'Street'])
test.loc[:, 'CentralAir'] = labelencoder.fit_transform(test.loc[:, 'CentralAir'])
train['tot_quality'] = train['BsmtQual_o'] + train['ExterQual_o'] + train['KitchenQual_o'] + train['GarageQual_o'] + train['FireplaceQu_o']
test['tot_quality'] = test['BsmtQual_o'] + test['ExterQual_o'] + test['KitchenQual_o'] + test['GarageQual_o'] + test['FireplaceQu_o']
numcols = train._get_numeric_data().columns
train_num = train.loc[:, numcols]
corr = train_num.corr(method='pearson')
(f, ax) = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.8, cbar_kws={'shrink': 0.5})
k = corr.unstack().sort_values().drop_duplicates()
k[(k > 0.8) | (k < -0.8)]
test.sort_values(by='Id', inplace=True)
test
train.sort_values(by='Id', inplace=True)
salePrice = train['SalePrice']
del train['logSalePrice']
del train['SalePrice']
train['flag'] = 'train'
test['flag'] = 'test'
all_data = pd.concat([train, test], ignore_index=True, sort=False)
from scipy.stats import skew
numeric_feats = all_data.dtypes[train.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
skewed_feats
numcols = all_data._get_numeric_data().columns
all_data_num = all_data.loc[:, numcols]
k = all_data_num.columns.to_series()[np.isinf(all_data_num).any()]
all_data[k] = all_data[k].replace(-np.inf, np.nan)
print(all_data.columns.values)
all_data_mod = all_data[['LotArea', 'GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'BsmtQual_o', 'ExterQual_o', 'GarageCars', '1stFlrSF', '2ndFlrSF', 'FullBath', 'FireplaceQu_o', 'KitchenQual_o', 'MSZoning_o', 'PavedDrive_o', 'LotShape_o', 'Utilities_o', 'LandContour_o', 'Functional_o', 'tot_bath', 'bed_bath_kitch', 'area_floors', 'GarageArea', 'Fireplaces', 'BedroomAbvGr', 'tot_quality', 'sale_built_yr', 'remod_built_yr', 'MSSubClass', 'neighborhood_flag', 'MasVnrArea', 'garage_area_per_car', 'floor_by_lot', 'remod_flag', 'new_flag', 'unf_bsmt', 'bsmt_by_total', 'bsmt_bath', 'flag', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF']]
all_data_mod
all_data_mod = all_data[['LotArea', 'GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'BsmtQual_o', 'ExterQual_o', 'GarageCars', '1stFlrSF', '2ndFlrSF', 'FullBath', 'FireplaceQu_o', 'KitchenQual_o', 'MSZoning_o', 'PavedDrive_o', 'LotShape_o', 'Utilities_o', 'LandContour_o', 'Functional_o', 'tot_bath', 'bed_bath_kitch', 'area_floors', 'GarageArea', 'Fireplaces', 'BedroomAbvGr', 'tot_quality', 'sale_built_yr', 'remod_built_yr', 'MSSubClass', 'neighborhood_flag', 'MasVnrArea', 'floor_by_lot', 'remod_flag', 'new_flag', 'unf_bsmt', 'bsmt_by_total', 'bsmt_bath', 'flag', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']]
train_mod = all_data_mod.loc[all_data_mod['flag'] == 'train', :]
test_mod = all_data_mod.loc[all_data_mod['flag'] == 'test', :]
del train_mod['flag']
del test_mod['flag']
my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
train_imp = my_imputer.fit_transform(train_mod)
test_imp = my_imputer.fit_transform(test_mod)
X = train_imp
y = np.log1p(salePrice)
(X_train, X_test, Y_train, Y_Test) = train_test_split(X, y, test_size=0.3, random_state=1)
regressor = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=10, max_features=10, min_samples_leaf=5)
feature_list = train_mod.columns