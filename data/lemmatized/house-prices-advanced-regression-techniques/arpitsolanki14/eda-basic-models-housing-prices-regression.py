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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
_input1.describe()
_input1['logSalePrice'] = np.log(_input1.SalePrice)
fig = px.histogram(_input1, x='SalePrice', title='Distribution of SalePrice', height=400)
fig.show()
fig1 = px.violin(_input1, x='SalePrice', title='Violin Plot for SalePrice', height=300)
fig1.update_traces(box_visible=True, meanline_visible=True)
fig1.show()
fig2 = px.violin(_input1, x='logSalePrice', title='Violin Plot for Log(SalePrice)', height=300)
fig2.update_traces(box_visible=True, meanline_visible=True)
fig2.show()
k = _input1.isnull().sum()
k[k > 0].sort_values(ascending=False)
numcols = _input1._get_numeric_data().columns
train_num = _input1.loc[:, numcols]
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
train_str_data = _input1.select_dtypes(include='object')
train_str_data['SalePrice'] = _input1['SalePrice']
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
fig = px.violin(_input1, y='SalePrice', x='Neighborhood', box=True, color='Neighborhood', title='Distribution of SalePrice by Neighborhood')
fig.update_layout(showlegend=False)
fig.show()
neighborhood_md = _input1.groupby('Neighborhood').agg({'SalePrice': 'median'}).reset_index().sort_values(by='SalePrice', ascending=False)
neighborhood_md
fig1 = px.bar(neighborhood_md, y='SalePrice', x='Neighborhood', color='SalePrice', title='Median SalePrice by Neighborhood')
fig1.update_layout(showlegend=False)
fig1.show()
condition = _input1.groupby(['Condition1', 'Condition2']).agg({'SalePrice': 'median', 'Utilities': 'count'}).reset_index().sort_values(by='SalePrice', ascending=False)
trans = condition.pivot(index='Condition1', columns='Condition2', values='SalePrice')
trans = trans.fillna(0)
cm = sns.light_palette('green', as_cmap=True)
s = trans.style.background_gradient(cmap=cm)
s
condition = condition.rename(columns={'SalePrice': 'avg_sale_price_cond', 'Utilities': 'total_records'})
condition = condition.sort_values(by='total_records', ascending=False, inplace=False)
condition
gp = _input1.groupby(['YrSold', 'MoSold']).agg({'SalePrice': 'median', 'LotArea': 'count'}).reset_index()
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
_input1 = pd.merge(_input1, neighborhood_md, left_on='Neighborhood', right_on='Neighborhood')
_input0 = pd.merge(_input0, neighborhood_md, left_on='Neighborhood', right_on='Neighborhood')
_input1['tot_bath'] = _input1['BsmtFullBath'] + 0.5 * _input1['BsmtHalfBath'] + _input1['FullBath'] + 0.5 * _input1['HalfBath']
_input1['bsmt_bath'] = _input1['BsmtFullBath'] + 0.5 * _input1['BsmtHalfBath']
_input1['bed_bath_kitch'] = _input1['tot_bath'] + _input1['BedroomAbvGr'] + _input1['KitchenAbvGr']
_input1['area_floors'] = _input1['1stFlrSF'] + _input1['2ndFlrSF'] + _input1['BsmtFinSF1'] + _input1['BsmtFinSF2']
_input1['bsmt_by_total'] = (_input1['BsmtFinSF1'] + _input1['BsmtFinSF2']) / _input1['area_floors']
_input1['unf_bsmt'] = (_input1['BsmtUnfSF'] / _input1['TotalBsmtSF']).fillna(0)
_input1['unf_bsmt'].replace([np.inf, -np.inf], 0)
_input1['porch_area_tot'] = _input1['OpenPorchSF'] + _input1['EnclosedPorch'] + _input1['3SsnPorch'] + _input1['ScreenPorch']
_input1['wood_deck_porch'] = (_input1['WoodDeckSF'] / _input1['porch_area_tot']).replace(np.inf, 0)
_input1['sale_built_yr'] = _input1['YrSold'] - _input1['YearBuilt']
_input1['remod_built_yr'] = _input1['YearRemodAdd'] - _input1['YearBuilt']
_input1['new_flag'] = 0
_input1.loc[_input1['YrSold'] - _input1['YearBuilt'], 'new_flag'] = 1
_input1['remod_flag'] = 0
_input1.loc[_input1['remod_built_yr'] >= 2, 'remod_flag'] = 1
_input1['floor_by_lot'] = _input1['area_floors'] / _input1['LotArea']
_input1['garage_area_per_car'] = (_input1['GarageArea'] / _input1['GarageCars']).fillna(0)
_input1['garage_area_per_car'] = _input1['garage_area_per_car'].replace([np.inf, -np.inf], 0)
_input0['tot_bath'] = _input0['BsmtFullBath'] + 0.5 * _input0['BsmtHalfBath'] + _input0['FullBath'] + 0.5 * _input0['HalfBath']
_input0['bsmt_bath'] = _input0['BsmtFullBath'] + 0.5 * _input0['BsmtHalfBath']
_input0['bed_bath_kitch'] = _input0['tot_bath'] + _input0['BedroomAbvGr'] + _input0['KitchenAbvGr']
_input0['area_floors'] = _input0['1stFlrSF'] + _input0['2ndFlrSF'] + _input0['BsmtFinSF1'] + _input0['BsmtFinSF2']
_input0['bsmt_by_total'] = (_input0['BsmtFinSF1'] + _input0['BsmtFinSF2']) / _input0['area_floors']
_input0['unf_bsmt'] = (_input0['BsmtUnfSF'] / _input0['TotalBsmtSF']).fillna(0)
_input0['unf_bsmt'].replace([np.inf, -np.inf], 0)
_input0['porch_area_tot'] = _input0['OpenPorchSF'] + _input0['EnclosedPorch'] + _input0['3SsnPorch'] + _input0['ScreenPorch']
_input0['wood_deck_porch'] = (_input0['WoodDeckSF'] / _input0['porch_area_tot']).replace(np.inf, 0)
_input0['sale_built_yr'] = _input0['YrSold'] - _input0['YearBuilt']
_input0['remod_built_yr'] = _input0['YearRemodAdd'] - _input0['YearBuilt']
_input0['new_flag'] = 0
_input0.loc[_input0['YrSold'] - _input0['YearBuilt'] == 0, 'new_flag'] = 1
_input0['remod_flag'] = 0
_input0.loc[_input0['remod_built_yr'] >= 2, 'remod_flag'] = 1
_input0['floor_by_lot'] = _input0['area_floors'] / _input0['LotArea']
_input0['garage_area_per_car'] = (_input0['GarageArea'] / _input0['GarageCars']).fillna(0)
_input0['garage_area_per_car'] = _input0['garage_area_per_car'].replace([np.inf, -np.inf], 0)
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
_input1 = mapping_var(mapping, _input1, l_col)
_input1 = mapping_var(mapping_ms_zoning, _input1, l_col_zon)
_input1 = mapping_var(mapping_paved_Drive, _input1, l_col_pav)
_input1 = mapping_var(mapping_shape, _input1, l_col_shape)
_input1 = mapping_var(mapping_utilities, _input1, l_col_util)
_input1 = mapping_var(mapping_contour, _input1, l_col_con)
_input1 = mapping_var(mapping_functional, _input1, l_col_fun)
_input0 = mapping_var(mapping, _input0, l_col)
_input0 = mapping_var(mapping_ms_zoning, _input0, l_col_zon)
_input0 = mapping_var(mapping_paved_Drive, _input0, l_col_pav)
_input0 = mapping_var(mapping_shape, _input0, l_col_shape)
_input0 = mapping_var(mapping_utilities, _input0, l_col_util)
_input0 = mapping_var(mapping_contour, _input0, l_col_con)
_input0 = mapping_var(mapping_functional, _input0, l_col_fun)
labelencoder = LabelEncoder()
_input1.loc[:, 'Street'] = labelencoder.fit_transform(_input1.loc[:, 'Street'])
_input1.loc[:, 'CentralAir'] = labelencoder.fit_transform(_input1.loc[:, 'CentralAir'])
_input0.loc[:, 'Street'] = labelencoder.fit_transform(_input0.loc[:, 'Street'])
_input0.loc[:, 'CentralAir'] = labelencoder.fit_transform(_input0.loc[:, 'CentralAir'])
_input1['tot_quality'] = _input1['BsmtQual_o'] + _input1['ExterQual_o'] + _input1['KitchenQual_o'] + _input1['GarageQual_o'] + _input1['FireplaceQu_o']
_input0['tot_quality'] = _input0['BsmtQual_o'] + _input0['ExterQual_o'] + _input0['KitchenQual_o'] + _input0['GarageQual_o'] + _input0['FireplaceQu_o']
numcols = _input1._get_numeric_data().columns
train_num = _input1.loc[:, numcols]
corr = train_num.corr(method='pearson')
(f, ax) = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.8, cbar_kws={'shrink': 0.5})
k = corr.unstack().sort_values().drop_duplicates()
k[(k > 0.8) | (k < -0.8)]
_input0 = _input0.sort_values(by='Id', inplace=False)
_input0
_input1 = _input1.sort_values(by='Id', inplace=False)
salePrice = _input1['SalePrice']
del _input1['logSalePrice']
del _input1['SalePrice']
_input1['flag'] = 'train'
_input0['flag'] = 'test'
all_data = pd.concat([_input1, _input0], ignore_index=True, sort=False)
from scipy.stats import skew
numeric_feats = all_data.dtypes[_input1.dtypes != 'object'].index
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