import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sb
sb.set()
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import string
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_len = len(_input1)
data_all = _input1.append(_input0, sort=False, ignore_index=True)
features = data_all.shape[1]
idUnique = len(set(data_all.Id))
idTotal = data_all.shape[0]
idDupli = idTotal - idUnique
print('There are ' + str(idDupli) + ' duplicate IDs for ' + str(idTotal) + ' total entries')
print('There are ' + str(features) + ' variables for ' + str(idTotal) + ' total entries')
quantity = [f for f in data_all.columns if data_all.dtypes[f] != 'object']
quantity.remove('SalePrice')
quantity.remove('Id')
quality = [f for f in data_all.columns if data_all.dtypes[f] == 'object']
print(quantity)
print(quality)
missing_quant = (data_all[quantity].isnull().sum() / data_all[quantity].isnull().count()).sort_values(ascending=False)
missing_quant = missing_quant[missing_quant > 0] * 100
print('There are {} quantitative features with  missing values :'.format(missing_quant.shape[0]))
missing_quant = pd.DataFrame({'Percent': missing_quant})
missing_quant.head()
np.where(pd.isnull(data_all.GarageArea))
data_all['LotFrontage'] = data_all.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))
data_all.at[2576, 'GarageYrBlt'] = 0
data_all['GarageYrBlt'] = data_all.groupby(['Neighborhood'])['GarageYrBlt'].apply(lambda x: x.fillna(x.median()))
for col in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:
    data_all[col] = data_all[col].fillna(0)
missing_qual = (data_all[quality].isnull().sum() / data_all[quality].isnull().count()).sort_values(ascending=False)
missing_qual = missing_qual[missing_qual > 0] * 100
print('There are {} qualitative features with  missing values :'.format(missing_qual.shape[0]))
missing_qual = pd.DataFrame({'Percent': missing_qual})
missing_qual.head(10)
data_all['MSZoning'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['MSZoning'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['Utilities'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['Utilities'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['Exterior1st'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['Exterior1st'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['Exterior2nd'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['Exterior2nd'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['MasVnrType'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['MasVnrType'].apply(lambda x: x.fillna(x.value_counts().index[0]))
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'Alley', 'Fence', 'PoolQC', 'MiscFeature']:
    data_all[col] = data_all[col].fillna('None')
data_all['Electrical'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['Electrical'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['KitchenQual'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['KitchenQual'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['Functional'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['Functional'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all['SaleType'] = data_all.groupby(['Neighborhood', 'MSSubClass'])['SaleType'].apply(lambda x: x.fillna(x.value_counts().index[0]))
data_all.iloc[np.where(data_all.GrLivArea > 4000)]
scatter = sb.regplot(x='GrLivArea', y='SalePrice', fit_reg=False, data=data_all)
data_all = data_all.drop(data_all[data_all['Id'] == 524].index)
data_all = data_all.drop(data_all[data_all['Id'] == 1299].index)
train_len = train_len - 2
data_all.loc[2549, 'GrLivArea'] = data_all['GrLivArea'].median()
data_all.loc[2549, 'LotArea'] = data_all['LotArea'].median()
corrmatrix = data_all[:train_len].corr()
(f, ax) = plt.subplots(figsize=(30, 24))
k = 36
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_all[:train_len][cols].values.T)
sb.set(font_scale=1.0)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 18}, yticklabels=cols.values, xticklabels=cols.values)
quantitative = [f for f in data_all.columns if f in cols]
quantitative.remove('SalePrice')
varx = pd.melt(data_all, id_vars=['SalePrice'], value_vars=quantitative)
gx = sb.FacetGrid(varx, col='variable', col_wrap=3, sharex=False, sharey=False, height=5)
gx = gx.map(sb.regplot, 'value', 'SalePrice')
data_all['KitchenQual'] = data_all['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa'], [4, 3, 2, 1], inplace=False)
data_all['FireplaceQu'] = data_all['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], [6, 5, 4, 3, 2, 1], inplace=False)
data_all['GarageQual'] = data_all['GarageQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], [6, 5, 4, 3, 2, 1], inplace=False)
data_all['GarageCond'] = data_all['GarageCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], [6, 5, 4, 3, 2, 1], inplace=False)
data_all['PoolQC'] = data_all['PoolQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'None'], [5, 4, 3, 2, 1], inplace=False)
data_all['ExterQual'] = data_all['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa'], [4, 3, 2, 1], inplace=False)
data_all['ExterCond'] = data_all['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1], inplace=False)
data_all['BsmtQual'] = data_all['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], [6, 5, 4, 3, 2, 1], inplace=False)
data_all['BsmtCond'] = data_all['BsmtCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], [6, 5, 4, 3, 2, 1], inplace=False)
data_all['BsmtExposure'] = data_all['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', 'None'], [5, 4, 3, 2, 1], inplace=False)
data_all['HeatingQC'] = data_all['HeatingQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1], inplace=False)
data_all['MSSubClass'] = data_all['MSSubClass'].astype(str)
data_all['YrSold'] = data_all['YrSold'].astype(str)
data_all['MoSold'] = data_all['MoSold'].astype(str)
data_all['GarageScale'] = data_all['GarageCars'] * data_all['GarageArea']
data_all['GarageOrdinal'] = data_all['GarageQual'] + data_all['GarageCond']
data_all['AllPorch'] = data_all['OpenPorchSF'] + data_all['EnclosedPorch'] + data_all['3SsnPorch'] + data_all['ScreenPorch']
data_all['ExterOrdinal'] = data_all['ExterQual'] + data_all['ExterCond']
data_all['KitchenCombined'] = data_all['KitchenQual'] * data_all['KitchenAbvGr']
data_all['FireplaceCombined'] = data_all['FireplaceQu'] * data_all['Fireplaces']
data_all['BsmtOrdinal'] = data_all['BsmtQual'] + data_all['BsmtCond']
data_all['BsmtFinishedAll'] = data_all['BsmtFinSF1'] + data_all['BsmtFinSF2']
data_all['AllFlrSF'] = data_all['1stFlrSF'] + data_all['2ndFlrSF']
data_all['OverallCombined'] = data_all['OverallQual'] + data_all['OverallCond']
data_all['TotalFullBath'] = data_all['BsmtFullBath'] + +data_all['FullBath']
data_all['TotalHalfBath'] = data_all['HalfBath'] + data_all['BsmtHalfBath']
data_all['TotalSF'] = data_all['AllFlrSF'] + data_all['TotalBsmtSF']
data_all['YrBltAndRemod'] = data_all['YearRemodAdd'] + data_all['YearBuilt']
cat_features = [f for f in data_all.columns if data_all.dtypes[f] == 'object']
categorical = []
for i in data_all.columns:
    if i in cat_features:
        categorical.append(i)
data_cat = data_all[categorical]
data_cat = pd.get_dummies(data_cat)
overfit = []
for i in data_cat.columns:
    counts = data_cat[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(data_cat) * 100 > 99.95:
        overfit.append(i)
data_cat = data_cat.drop(overfit, axis=1)
print('There are {} qualitative features with zeros > 99.95 :'.format(len(overfit)))
numeric = [f for f in data_all.columns if data_all.dtypes[f] != 'object']
numeric.remove('Id')
features = numeric
spr = pd.DataFrame()
spr['feature'] = features
spr['spearman'] = [data_all[f].corr(data_all['SalePrice'], 'spearman') for f in features]
spr = spr.sort_values('spearman')
numeric_cols = spr['feature'].tail(47)
plt.figure(figsize=(6, 0.25 * len(features)))
bar = sb.barplot(data=spr, y='feature', x='spearman', orient='h')
data_all[:train_len].loc[:, 'SalePrice'] = np.log1p(data_all[:train_len]['SalePrice'])
fig = plt.figure()
res = stats.probplot(data_all[:train_len]['SalePrice'], plot=plt)
numerical = numeric_cols.values.tolist()
numerical.remove('SalePrice')
data_num = data_all[numerical].astype(float)
skewness = data_num.apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]
print('There are {} numerical features with absolute Skew > 0.5 :'.format(skewness.shape[0]))
skewness = pd.DataFrame({'Skew': skewness})
skewness.head(10)
for i in skewness.index.tolist():
    data_num[i] = boxcox1p(data_num[i], boxcox_normmax(data_num[i] + 1))
skewness = data_num.apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]
neg_skew = skewness[skewness < 0]
data_num.loc[:, 'GarageOrdinal'] = data_num['GarageOrdinal'] ** 1.1
data_num.loc[:, 'GarageQual'] = data_num['GarageQual'] ** 1.2
data_num.loc[:, 'GarageCond'] = data_num['GarageCond'] ** 1.3
data_num.loc[:, 'HeatingQC'] = data_num['HeatingQC'] ** 1.6
pos_skew = skewness[skewness > 0]
pos_skew1 = ['Fireplaces', 'BsmtFullBath', 'OpenPorchSF', 'FireplaceCombined', 'TotalHalfBath', 'WoodDeckSF', '2ndFlrSF']
data_num.loc[:, pos_skew1] = np.sqrt(data_num[pos_skew1])
data_num.loc[:, 'HalfBath'] = np.cbrt(data_num['HalfBath'])
data_num.loc[:, 'MasVnrArea'] = np.log2(data_num['MasVnrArea'] + 1)
skewness = data_num.apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[abs(skewness) > 1.0]
data_num = data_num.drop(skewness.index.tolist(), axis=1)
print('There are {} numerical features with absolute Skew > 1.0 :'.format(skewness.shape[0]))
scatter = sb.regplot(x='GrLivArea', y='SalePrice', fit_reg=False, data=data_all)
data_num = (data_num - data_num.mean()) / data_num.std()
df_all = pd.concat([data_num, data_cat], axis=1)
features = data_cat.columns.append(data_num.columns)
df_train = df_all[:1458]
df_test = df_all[1458:]
target = 'SalePrice'
X = df_train[features]
y = data_all[:1458][target]
X_test = df_test[features]

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    return rmse
ridge = Ridge(alpha=14)
cv_ridge = rmse_cv(ridge)