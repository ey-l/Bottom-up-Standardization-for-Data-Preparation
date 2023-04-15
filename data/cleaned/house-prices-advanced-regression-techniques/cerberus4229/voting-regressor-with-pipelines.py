import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_df.head()
train_df.shape
sns.set_style('darkgrid')
sns.set_color_codes(palette='dark')
(f, ax) = plt.subplots(figsize=(9, 9))
sns.distplot(train_df['SalePrice'], color='m', axlabel='SalePrice')
ax.set(title='Histogram for SalePrice')

corr_mat = train_df.corr()
plt.subplots(figsize=(12, 10))
sns.heatmap(corr_mat, square=True, robust=True, cmap='OrRd', cbar_kws={'fraction': 0.01}, linewidth=1)
k = 10
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
cmap_ch = sns.cubehelix_palette(as_cmap=True, light=0.95)
hm = sns.heatmap(cm, cmap=cmap_ch, cbar=True, annot=True, square=True, robust=True, cbar_kws={'fraction': 0.01}, annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values, linewidth=1)

chart = sns.catplot(data=train_df, x='OverallQual', y='SalePrice', kind='box', height=8, palette='Set2')
chart.set_xticklabels(fontweight='light', fontsize='large')
chart = sns.catplot(data=train_df, x='YearBuilt', y='SalePrice', kind='box', height=5, aspect=4, palette='Set2')
chart.set_xticklabels(fontweight='light', fontsize='large', rotation=90, horizontalalignment='center')
plt.figure(figsize=(10, 8))
sns.scatterplot(data=train_df, x='TotalBsmtSF', y='SalePrice', alpha=0.65, color='g')
plt.figure(figsize=(10, 8))
sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice', alpha=0.65, color='b')
chart = sns.catplot(data=train_df, x='GarageCars', y='SalePrice', kind='box', height=6, palette='Set2')
chart.set_xticklabels(fontweight='light', fontsize='large')
train_df_IDs = train_df['Id']
test_df_IDs = test_df['Id']
train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)
print(train_df.shape)
print(test_df.shape)
train_df['SalePrice_log'] = np.log(train_df['SalePrice'])
sns.distplot(train_df['SalePrice_log'], color='m', axlabel='SalePrice_log')
train_df = train_df.drop('SalePrice_log', axis=1)
train_df.drop(train_df[(train_df['GrLivArea'] > 4500) & (train_df['SalePrice'] < 300000)].index, inplace=True)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice', alpha=0.65, color='b')
housing = train_df.drop('SalePrice', axis=1)
housing_labels = train_df['SalePrice']
total_series = housing.isnull().sum().sort_values(ascending=False)
perc_series = (housing.isnull().sum() / housing.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_series, perc_series * 100], axis=1, keys=['Total #', 'Percent'])
missing_data.head(20)
cols_int_to_str = ['MSSubClass', 'YrSold', 'MoSold', 'GarageYrBlt']
for col in cols_int_to_str:
    housing[col] = housing[col].astype(str)
    test_df[col] = test_df[col].astype(str)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
housing_num = housing.select_dtypes(include=numerics)
print(housing_num.shape)
cat_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
housing_cat_none = housing[cat_none]
housing_cat_freq = housing[housing.columns.difference(cat_none) & housing.columns.difference(housing_num.columns)]
from sklearn.base import BaseEstimator, TransformerMixin
(BsmtFinSF1_ix, BsmtFinSF2_ix, flr_1_ix, flr_2_ix, FullBath_ix, HalfBath_ix, BsmtFullBath_ix, BsmtHalfBath_ix, OpenPorchSF_ix, SsnPorch_ix, EnclosedPorch_ix, ScreenPorch_ix, WoodDeckSF_ix) = [list(housing_num.columns).index(col) for col in ('BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF')]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, Total_sqr_footage=True, Total_Bathrooms=True, Total_porch_sf=True):
        self.Total_sqr_footage = Total_sqr_footage
        self.Total_Bathrooms = Total_Bathrooms
        self.Total_porch_sf = Total_porch_sf

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.Total_sqr_footage:
            Total_sqr_footage = X[:, BsmtFinSF1_ix] + X[:, BsmtFinSF2_ix] + X[:, flr_1_ix] + X[:, flr_2_ix]
        if self.Total_Bathrooms:
            Total_Bathrooms = X[:, FullBath_ix] + X[:, HalfBath_ix] + X[:, BsmtFullBath_ix] + X[:, BsmtHalfBath_ix]
        if self.Total_porch_sf:
            Total_porch_sf = X[:, OpenPorchSF_ix] + X[:, SsnPorch_ix] + X[:, EnclosedPorch_ix] + X[:, ScreenPorch_ix] + X[:, WoodDeckSF_ix]
        return np.c_[X, Total_sqr_footage, Total_Bathrooms, Total_porch_sf]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('attribs_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler())])
from sklearn.preprocessing import OneHotEncoder
cat_pipeline_none = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='None')), ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
cat_pipeline_freq = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer(transformers=[('num', num_pipeline, housing_num.columns), ('cat_none', cat_pipeline_none, housing_cat_none.columns), ('cat_freq', cat_pipeline_freq, housing_cat_freq.columns)])