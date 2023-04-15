from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np

def normal(mean, std, color='black', ax=None):
    x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
    p = stats.norm.pdf(x, mean, std)
    if ax is None:
        plt.plot(x, p, color, linewidth=2)
    else:
        ax.plot(x, p, color, linewidth=2)
data = pd.concat([pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv'), pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')], axis=0, sort=False)
test_id = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')['Id']
data.drop(columns='Id', inplace=True)
data[:1460].info()
data[:1460].head()
plt.figure(figsize=(30, 25))
sns.heatmap(data[:1460].corr(), cmap='plasma', annot=True)
data[:1460].corr()['SalePrice']
(_, ax) = plt.subplots(4, 3, figsize=(50, 60))
for (i, feature) in enumerate(['OverallQual', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']):
    sns.stripplot(data=data[:1460], x=feature, y='SalePrice', ax=ax[i, 0])
    sns.violinplot(data=data[:1460], x=feature, y='SalePrice', ax=ax[i, 1])
    sns.boxplot(data=data[:1460], x=feature, y='SalePrice', ax=ax[i, 2])

(_, ax) = plt.subplots(3, 2, figsize=(50, 60))
ax = ax.flatten()
for (i, feature) in enumerate(['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']):
    sns.regplot(data=data[:1460], x=feature, y='SalePrice', scatter_kws={'alpha': 0.3}, ax=ax[i])
    ax[i].set_title(f'{feature} VS. SalePrice')
data['MSSubClass'] = data['MSSubClass'].apply(str)
data['YrSold'] = data['YrSold'].apply(str)
data['MoSold'] = data['MoSold'].apply(str)
data['TotalQual'] = data['OverallQual'] + data['OverallCond']
data['TotalBath'] = data['FullBath'] + data['BsmtFullBath'].fillna(0) + 0.5 * data['HalfBath'] + 0.5 * data['BsmtHalfBath'].fillna(0)
data['GarageAreaPerCar'] = (data['GarageArea'] / data['GarageCars']).fillna(0)
data['TimeTakenToBuildGarage'] = (data['GarageYrBlt'] - data['YearBuilt']).fillna(0)
data['AreaPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['KitchenAbvGr'])
data['TotalSF'] = data['1stFlrSF'] + data['TotalBsmtSF'].fillna(0) + data['2ndFlrSF'] + data['GrLivArea']
data['HighQualSF'] = data['GrLivArea'] + data['1stFlrSF'] + data['2ndFlrSF'] + 0.25 * data['GarageArea'].fillna(0) + 0.5 * data['TotalBsmtSF'].fillna(0) + data['MasVnrArea'].fillna(0)
for feature in data[:1460].columns.values:
    if data[:1460][feature].dtype in ['int64', 'float64']:
        print('Skewness of {}: {:.2f}'.format(feature, data[:1460][feature].skew()))
for feature in ['Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Utilities']:
    data[feature].fillna(data[feature].mode()[0])
for feature in [column for column in data.columns.values if data[column].dtype == 'object']:
    dummies = pd.get_dummies(data[feature], prefix=feature)
    data.drop(columns=[feature], inplace=True)
    data = pd.concat([data, dummies], axis=1)
    print(f'Dummy variables created for {feature}')
train_data = data[:1460]
test_data = data[1460:]
zero_imputer = Pipeline([('imputing', SimpleImputer(strategy='constant', fill_value=0))])
median_imputer = Pipeline([('imputing', SimpleImputer(strategy='median'))])
normal_transform_pipeline = Pipeline([('transforming', QuantileTransformer(output_distribution='normal'))])
uniform_transform_pipeline = Pipeline([('transforming', QuantileTransformer(output_distribution='uniform'))])
log_transform_pipeline = Pipeline([('transforming', FunctionTransformer(np.log1p))])
numeric_features = [feature for feature in train_data.columns.values if train_data[feature].dtype in ['int64', 'float64']]
skewed_columns = [feature for feature in numeric_features if round(abs(train_data[feature].skew())) > 0.5 and feature != 'SalePrice']
data_preprocessor = ColumnTransformer([('median_imputer', median_imputer, ['LotFrontage']), ('zero_imputer', zero_imputer, ['GarageCars', 'GarageArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'HalfBath', 'GarageCars', 'GarageArea', 'GarageYrBlt']), ('normal_transformer', normal_transform_pipeline, ['LotArea', 'GrLivArea', 'HighQualSF', 'AreaPerRoom', 'TotalSF']), ('uniform_transformer', uniform_transform_pipeline, ['YearBuilt', 'YearRemodAdd', 'BsmtUnfSF', 'TotalBsmtSF']), ('log_transformer', log_transform_pipeline, skewed_columns)], remainder='passthrough')