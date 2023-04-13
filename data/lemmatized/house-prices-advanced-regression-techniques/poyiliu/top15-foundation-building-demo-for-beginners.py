import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.head())
print(_input0.head())
print('train shape', _input1.shape)
print('test shape', _input0.shape)
print('duplicated rows', _input1.duplicated().sum())
print('columns containing missing values', _input1.isnull().any().sum())
_input0['SalePrice'] = np.nan
data_all = pd.concat([_input1, _input0], ignore_index=True)
print('merged shape ', data_all.shape)
print(data_all.info())
data_object = data_all.select_dtypes('object')
print('object shape ', data_object.shape)
data_num = data_all.select_dtypes(['int64', 'float64'])
print('num shape ', data_num.shape)
data_num.describe()
for i in data_object.columns:
    print(data_object[i].value_counts())
int_col = _input1.select_dtypes('int64').columns

def show_sample(column_list, row_range_start=0):
    i = 0
    while i < len(column_list):
        try:
            print(_input1[column_list[i:i + 10]].iloc[row_range_start:row_range_start + 10])
            i += 10
        except:
            print(_input1[column_list[i:]].iloc[row_range_start:row_range_start + 10])
show_sample(int_col)
discrete_int = ['Id', 'MSSubClass']
time_int = ['YrSold', 'YearBuilt', 'YearRemodAdd', 'MoSold']
_input1[discrete_int] = _input1[discrete_int].astype('object')
data_all[discrete_int] = data_all[discrete_int].astype('object')

def get_num_features(df):
    num = list(df.select_dtypes(['int64', 'float64']).columns)
    try:
        num.remove('SalePrice')
    except:
        pass
    return num

def get_cat_features(df):
    return list(df.select_dtypes('object').columns)
num_train = _input1.select_dtypes(['int64', 'float64'])
object_train = _input1.select_dtypes('object')
num_train_corr = num_train.corr()
num_all = data_all.select_dtypes(['int64', 'float64'])
object_all = data_all.select_dtypes('object')
num_all_corr = num_all.corr()
(fig, ax) = plt.subplots(figsize=(20, 20))
sns.heatmap(num_all_corr, cmap='Reds')
(fig, ax) = plt.subplots(figsize=(20, 20))
sns.heatmap(num_train_corr, cmap='Reds')
Correlation = pd.DataFrame(num_train.corr()['SalePrice'])
Correlation['Abs'] = np.abs(Correlation['SalePrice'])
Correlation = Correlation.sort_values(by='Abs', ascending=False)
important_features_CC = list(Correlation[Correlation['Abs'] > 0.5].index)
important_features_CC.remove('SalePrice')
print(important_features_CC)
(fig, ax) = plt.subplots(figsize=(20, 20))
sns.set(font_scale=1.5)
sns.heatmap(_input1[important_features_CC + ['SalePrice']].corr(), annot=True, annot_kws={'size': 20})
data_all['GarageCars'].describe()
data_all['FullBath'].describe()
data_all['TotRmsAbvGrd'].describe()
data_all['YearBuilt'].describe()
data_all['YearRemodAdd'].describe()
object_train['SalePrice'] = _input1['SalePrice']
sns.set(font_scale=1)
plt.rcParams['figure.figsize'] = (10, 6)
for i in object_train.columns:
    if i in ['Id', 'SalePrice']:
        pass
    else:
        categories = object_train[i].unique()
        print('Categories for', i, ':', len(categories))
        sns.countplot(x=i, data=object_train)
        plt.title(i)
        for j in categories:
            plt.hist(object_train[object_train[i] == j]['SalePrice'], alpha=0.5, label=j)
        plt.legend(loc='upper right')
sns.displot(_input1['SalePrice'])

def draw_time(data, time_feature, y='SalePrice'):
    frame_mean = data.groupby(time_feature)[y].mean()
    frame_count = data.groupby(time_feature)[y].count()
    sns.lineplot(x=frame_mean.index, y=frame_mean)
    plt.title('Mean ' + y + ' Against ' + time_feature)
    sns.lineplot(x=frame_count.index, y=frame_count)
    plt.title('Count ' + y + ' Against ' + time_feature)
data_all['Sold_time'] = data_all['YrSold'].astype(str) + '/' + data_all['MoSold'].astype(str)
data_all['Sold_time'] = pd.to_datetime(data_all['Sold_time'], format='%Y/%m')
data_all['MoSold'] = data_all['MoSold'].astype('object')
draw_time(data=data_all, time_feature='Sold_time')
draw_time(data=data_all, time_feature='YearBuilt')
draw_time(data=data_all, time_feature='YearRemodAdd')
data_all['ExterQual'] = data_all['ExterQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data_all['ExterCond'] = data_all['ExterCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data_all['BsmtQual'] = data_all['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data_all['BsmtCond'] = data_all['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data_all['BsmtExposure'] = data_all['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
data_all['HeatingQC'] = data_all['HeatingQC'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data_all['KitchenQual'] = data_all['KitchenQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
data_all['FireplaceQu'] = data_all['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data_all['GarageFinish'] = data_all['GarageFinish'].replace({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
data_all['GarageQual'] = data_all['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data_all['GarageCond'] = data_all['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
data_all['PavedDrive'] = data_all['PavedDrive'].replace({'Y': 3, 'P': 2, 'N': 1})
data_all['PoolQC'] = data_all['PoolQC'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0})
data_all['Fence'] = data_all['Fence'].replace({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
high_card_col_cat = [i for i in get_cat_features(data_all) if len(data_all[i].unique()) >= 10]
high_card_col_cat.remove('Id')
from scipy import stats
stats.pearsonr(data_all['ExterQual'], data_all['ExterCond'])
data_all['OverallValue'] = data_all['OverallQual'] * data_all['OverallCond']
data_all['ExterValue'] = data_all['ExterQual'] * data_all['ExterCond']
data_all['BsmtQual'] = data_all['BsmtQual'] * data_all['BsmtCond']
data_all['GarageValue'] = data_all['GarageQual'] * data_all['GarageCond']
data_all['TotalArea'] = data_all['TotalBsmtSF'] + data_all['1stFlrSF'] + data_all['2ndFlrSF']
print(data_all.groupby('MSSubClass')['SalePrice'].mean())
print(data_all.groupby('Exterior1st')['SalePrice'].mean())
print(data_all.groupby('Exterior2nd')['SalePrice'].mean())
MSSubClass_Stories = {'20': 1, '30': 1, '40': 1, '45': 1.5, '50': 1.5, '60': 2, '70': 2, '75': 2.5, '120': 1, '150': 1.5, '160': 2}
MSSubClass_Ages = {'20': 1, '30': 0, '40': 0.5, '45': 0.5, '50': 0.5, '60': 1, '70': 0, '75': 0.5, '90': 0.5, '120': 1, '150': 0.5, '160': 1}
MSSubClass_Other = []

def get_MSSubClass_Stories(data):
    if str(data) in MSSubClass_Stories.keys():
        return MSSubClass_Stories[str(data)]
    else:
        return 2

def get_MSSubClass_Ages(data):
    if str(data) in MSSubClass_Ages.keys():
        return MSSubClass_Ages[str(data)]
    else:
        return 0.5
data_all['Stories'] = data_all['MSSubClass'].apply(get_MSSubClass_Stories)
data_all['Ages'] = data_all['MSSubClass'].apply(get_MSSubClass_Ages)
geo = {'North': ['Blmngtn', 'BrDale', 'ClearCr', 'Gilbert', 'Names', 'NoRidge', 'NPkVill', 'NoRidge', 'NridgHt', 'Sawyer', 'Somerst', 'StoneBr', 'Veenker', 'NridgHt'], 'South': ['Blueste', 'Edwards', 'Mitchel', 'MeadowV', 'SWISU', 'IDOTRR', 'Timber'], 'Downtown': ['BrkSide', 'Crawfor', 'OldTown', 'CollgCr'], 'West': ['Edwards', 'NWAmes', 'SWISU', 'SawyerW']}

def find_geo(neighborhood):
    for (key, value) in geo.items():
        if neighborhood in value:
            return key
        else:
            pass
    return np.nan
data_all['Geo'] = data_all['Neighborhood'].apply(find_geo)
print(data_all.groupby('Geo')['SalePrice'].mean())
from scipy.stats import skew
skewed_feats = data_all[get_num_features(data_all)].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew_col = skewed_feats[abs(skewed_feats) > 0.5].index
for i in high_skew_col:
    data_all[i] = np.log1p(data_all[i])
missing_counts = pd.DataFrame(data_all.isnull().sum().sort_values(ascending=False))
plt.figure(figsize=(50, 20))
sns.heatmap(data_all.isnull())
plt.figure(figsize=(20, 10))
missing_columns = missing_counts[missing_counts.iloc[:, 0] > 0]
sns.barplot(x=missing_columns.index, y=missing_columns.iloc[:, 0])
plt.xticks(rotation=90)
drop_col = list(missing_counts[missing_counts.iloc[:, 0] > 1000].index)
drop_col.remove('SalePrice')
missing_columns = missing_columns.drop(index='SalePrice')
try:
    data_all = data_all.drop(columns=drop_col, axis=0)
    missing_columns = missing_columns.drop(index=drop_col, axis=1)
except:
    pass
print(data_all[missing_columns.index].info())
missing_object = data_all[missing_columns.index].select_dtypes('object').columns
print('missing object', len(missing_object))
missing_num = data_all[missing_columns.index].select_dtypes(['int64', 'float64']).columns
print('missing num ', len(missing_num))
for i in missing_num:
    data_all[i] = data_all[i].fillna(data_all[i].median())
for j in missing_object:
    data_all[j] = data_all[j].fillna(data_all[j].mode()[0])
print(data_all.isnull().any().sum())
print(data_all[important_features_CC].info())
for i in _input1.columns:
    if len(_input1[i].unique()) < 20:
        sns.violinplot(x=_input1[i], y=_input1['SalePrice'])
    else:
        sns.scatterplot(x=_input1[i], y=_input1['SalePrice'])
extreme_ind = _input1[_input1['SalePrice'] > 700000].index
data_all = data_all.drop(index=extreme_ind, axis=1)
low_correl_col_num = list(Correlation[Correlation['Abs'] < 0.1].index)
try:
    low_correl_col_num.remove('MoSold')
except:
    pass
low_correl_col_cat = ['Street', 'LotShape', 'Utilities', 'LotConfig', 'LandSlope', 'RoofStyle']
data_all = data_all.drop(columns=low_correl_col_num + low_correl_col_cat + ['Id', 'Neighborhood', 'Sold_time', 'YrSold'])
all_num = get_num_features(data_all)
all_cat = get_cat_features(data_all)
for i in all_num:
    plt.hist(data_all[i])
    plt.title(i)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_col = ['YearRemodAdd', 'YearBuilt']
data_all[scale_col] = scaler.fit_transform(data_all[scale_col])
data_all['SalePrice'] = np.log(data_all['SalePrice'])
sns.displot(data_all['SalePrice'])
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
data_all_processed_x = pd.get_dummies(data_all)
data_all_processed_y = data_all['SalePrice']
y_train = data_all['SalePrice'].dropna()
X_train = data_all_processed_x[~data_all_processed_x['SalePrice'].isnull()].drop(columns='SalePrice')
X_test = data_all_processed_x[data_all_processed_x['SalePrice'].isnull()].drop(columns='SalePrice')
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=20210503)
print('X_train', X_train.shape)
print('X_val', X_val.shape)
print('X_test', X_test.shape)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
ElasticNet = ElasticNet(random_state=0, max_iter=5000)
parameters = {'alpha': [0.001, 0.0001, 1e-05]}
Grid = GridSearchCV(ElasticNet, parameters, cv=5, scoring='neg_root_mean_squared_error')