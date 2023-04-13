import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Id = _input0['Id']
_input1.info()
for col in list(_input0.columns):
    if _input1[col].dtype == 'object':
        if _input1[col].nunique() != _input0[col].nunique():
            print(col, ':', _input1[col].nunique(), 'uniques in train', _input0[col].nunique(), 'uniques in test')
q97 = _input1.SalePrice.quantile(0.99)
q003 = _input1.SalePrice.quantile(0.003)
print('Removed outliers, price values in: [{:.2f}; {:.2f}]'.format(0, q97))
_input1 = _input1[_input1.SalePrice < q97].reset_index()
Target = _input1.SalePrice
_input1 = _input1.drop('SalePrice', axis=1)
data = pd.concat([_input1, _input0], keys=['train', 'test'])
data = data.drop(['index', 'Id'], axis=1)
features = list(data.columns)
print('detected {:d} features'.format(len(features)))
Check_years = data.columns[data.columns.str.contains(pat='Year|Yr')]
data[Check_years.values].max().sort_values(ascending=False)
Replace_year = data.loc[data['GarageYrBlt'] > 2050, 'GarageYrBlt'].index.tolist()
data.loc[Replace_year, 'GarageYrBlt'] = data['GarageYrBlt'].mode()
Na_perc = []
for feature in features:
    percentage = len(data[feature][data[feature].isna() == True]) / len(data[feature])
    Na_perc.append(percentage)
Nas = pd.DataFrame({'feature': features, 'Na_perc': Na_perc}).sort_values(by='Na_perc', ascending=False)
Nas = Nas[Nas.Na_perc > 0.01]
plt.figure(figsize=(7, 7))
plt.title('NaN percentage')
plt.xlabel('Top features')
sns.barplot(x=Nas.Na_perc, y=Nas.feature, orient='h')
num_features = []
cat_features = []
ord_features = []
for feature in features:
    if str(data[feature].dtype) == 'object':
        data[feature] = data[feature].fillna('None')
        cat_features.append(feature)
        pass
    if str(data[feature].dtype) == 'float64':
        if data[feature].nunique() < 50:
            ord_features.append(feature)
        else:
            num_features.append(feature)
        pass
    if str(data[feature].dtype) == 'int64':
        if data[feature].nunique() < 50:
            ord_features.append(feature)
        else:
            num_features.append(feature)
        pass
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
num_ord_data = data[num_features + ord_features].copy()
scaler = RobustScaler()
num_ord_data = scaler.fit_transform(num_ord_data)
imputer = KNNImputer(n_neighbors=10, weights='uniform')
num_ord_data = imputer.fit_transform(num_ord_data)
num_ord_data_transformed = pd.DataFrame(num_ord_data, columns=num_features + ord_features)
trans_back = scaler.inverse_transform(num_ord_data_transformed)
num_ord_data_transformed = pd.DataFrame(trans_back, columns=num_features + ord_features)
data = data.reset_index()
for col in num_features + ord_features:
    data[col] = num_ord_data_transformed[col]
print('Numeric features:', len(num_features))
print('Ordinal features:', len(ord_features))
print('Categorical features:', len(cat_features))
if len(num_features) + len(ord_features) + len(cat_features) != len(features):
    print('missing features')
else:
    print('no missing,', len(num_features) + len(ord_features) + len(cat_features), 'features')
max_list_cat = []
max_list_ord = []
idx_max_cat = []
idx_max_ord = []
for feature in cat_features:
    cat_vals = data[feature].value_counts(normalize=True)
    max_list_cat.append(cat_vals.max())
    idx_max_cat.append(cat_vals.idxmax())
max_zero_percentage_num = []
for feature in num_features:
    percentage = data[feature][data[feature] == 0].count() / len(data[feature])
    max_zero_percentage_num.append(percentage)
max_zero_percentage_ord = []
for feature in ord_features:
    ord_vals = data[feature].value_counts(normalize=True)
    max_list_ord.append(ord_vals.max())
    idx_max_ord.append(ord_vals.idxmax())
zeroes_num = pd.DataFrame({'feature': num_features, 'ratio': max_zero_percentage_num}).sort_values(by='ratio', ascending=False)
vals_ord = pd.DataFrame({'feature': ord_features, 'ratio': max_list_ord, 'top_value': idx_max_ord}).sort_values(by='ratio', ascending=False)
singl_el = pd.DataFrame({'feature': cat_features, 'ratio': max_list_cat, 'top_value': idx_max_cat}).sort_values(by='ratio', ascending=False)
fig = plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title('Max zero ratio in numeric data')
plt.xlabel('Top features')
sns.barplot(x=zeroes_num.ratio, y=zeroes_num.feature, orient='h')
plt.subplot(1, 3, 2)
plt.title('Max zero ratio in numeric data')
plt.xlabel('Top features')
sns.barplot(x=vals_ord.ratio, y=vals_ord.feature, orient='h')
plt.subplot(1, 3, 3)
plt.title('Max single element ratio')
plt.xlabel('Top features')
sns.barplot(x=singl_el.ratio, y=singl_el.feature, orient='h')
plt.subplots_adjust(bottom=None, right=None, top=None, wspace=1, hspace=None)
data['TotalPorch'] = data['ScreenPorch'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
data['Rooms_kitchens'] = data['TotRmsAbvGrd'] + data['BsmtFullBath'] + data['BsmtHalfBath'] + data['FullBath'] + data['HalfBath']
data['Sqr_feet_per_room'] = (data['1stFlrSF'] + data['2ndFlrSF']) / data['TotRmsAbvGrd']
data['lotfr_lotarea'] = (data['LotFrontage'] + data['LotArea']) / 2
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['cond_qual'] = data['OverallCond'] * data['OverallQual']
data['HouseAge'] = data['YrSold'] - data['YearBuilt'] + 1
num_features.append('TotalSF')
num_features.append('TotalPorch')
num_features.append('lotfr_lotarea')
num_features.append('HouseAge')
ord_features.append('Rooms_kitchens')
ord_features.append('cond_qual')
imb_num_features = zeroes_num.feature.head(3)
imb_ord_features = vals_ord[['feature', 'top_value']].head(5)
imb_cat_features = singl_el.feature[singl_el.top_value == 'None'].head(16)
for feature in imb_num_features:
    data[feature][data[feature] != 0] = 0
    data[feature][data[feature] == 0] = 1
    num_features.remove(feature)
    cat_features.append(feature)
for feature in imb_ord_features.feature:
    data[feature][data[feature] != 0] = 0
    data[feature][data[feature] == 0] = 1
    ord_features.remove(feature)
    cat_features.append(feature)
for feature in imb_cat_features:
    data[feature][data[feature] != 'None'] = 'Present'
print('Numeric features:', len(num_features))
print('Ordinal features:', len(ord_features))
print('Categorical features:', len(cat_features))
if len(num_features) + len(ord_features) + len(cat_features) != len(features):
    print('missing features')
else:
    print('no missing,', len(num_features) + len(ord_features) + len(cat_features), 'features')
for feature in cat_features:
    ratio = data[feature][data[feature] == 'None'].count() / len(data[feature])
    if ratio > 0.75:
        cat_features.remove(feature)
        print('removed', feature)
for feature in ord_features:
    ratio = data[feature][data[feature] == 0].count() / len(data[feature])
    if ratio > 0.75:
        ord_features.remove(feature)
        print('removed', feature)
for col in num_features:
    data[col] = np.log(data[col] + 1)
for col in ord_features:
    data[col] = np.log(data[col] + 1)
_input1 = data[data.level_0 == 'train'].drop(['level_0', 'level_1'], axis=1)
_input0 = data[data.level_0 == 'test'].drop(['level_0', 'level_1'], axis=1)
Target.hist(bins=20)
Target = np.log(Target + 1)
Target.hist(bins=20)
from matplotlib import cm
a = 8
b = 3
c = 1
fig = plt.figure(figsize=(21, 35))
for feature in num_features:
    plt.subplot(a, b, c)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.scatter(_input1[feature], Target, s=2, label=feature, c=cm.cool(Target / Target.max() / 2))
    c = c + 1
a = 5
b = 3
c = 1
fig = plt.figure(figsize=(20, 25))
for feature in ord_features:
    plt.subplot(a, b, c)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.scatter(_input1[feature], Target, s=2, label=feature, c=cm.cool(Target / Target.max() / 2))
    c = c + 1
_input1['SalePrice'] = Target
(fig, ax) = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(_input1[num_features].corr(), dtype=bool))
sns.heatmap(_input1[num_features].corr(), ax=ax, cmap=cmap, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, mask=mask)
num_features.remove('GarageYrBlt')
num_features.remove('1stFlrSF')
ord_features.remove('MoSold')
ord_features.remove('YrSold')
num_list = []
ord_list = []
for feature in num_features:
    num_list.append(_input1[feature].corr(_input1.SalePrice))
numcorrs = pd.DataFrame({'num_feature': num_features, 'corr': num_list}).sort_values(by='corr', ascending=False)
for feature in ord_features:
    ord_list.append(_input1[feature].corr(_input1.SalePrice))
ordcorrs = pd.DataFrame({'ord_feature': ord_features, 'corr': ord_list}).sort_values(by='corr', ascending=False)
fig = plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.title('Correlation between numerical features and SalePrice')
sns.barplot(x=numcorrs['corr'], y=numcorrs['num_feature'], orient='h')
plt.subplot(1, 2, 2)
plt.title('Correlation between ordinal features and SalePrice')
sns.barplot(x=ordcorrs['corr'], y=ordcorrs['ord_feature'], orient='h')
plt.subplots_adjust(bottom=None, right=None, top=None, wspace=1, hspace=None)
from sklearn.linear_model import Lasso
y_train = _input1['SalePrice']
_input1.drop('SalePrice', axis=1)
data_train_dum = pd.get_dummies(_input1)
lasso = Lasso()