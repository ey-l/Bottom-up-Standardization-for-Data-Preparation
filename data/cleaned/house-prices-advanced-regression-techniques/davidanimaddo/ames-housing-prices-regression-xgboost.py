import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_log_error, r2_score
from statistics import mean, mode, median
import statsmodels.api as sma
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rcParams
from platform import python_version
print('Python Version: ', python_version())
RANDOM_SEED = 123
rcParams['figure.figsize'] = (10, 6)
sns.set_theme(palette='muted', style='whitegrid')
path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
df = pd.read_csv(path)
print(df.shape)
df.head()
path_test = '_data/input/house-prices-advanced-regression-techniques/test.csv'
df_test = pd.read_csv(path_test)
print(df_test.shape)
df_test.head()
print(df.dtypes.value_counts(), end='\n' * 2)
print(df_test.dtypes.value_counts())
df['SalePrice'].isnull().any()
df['SalePrice'].describe()
print(f'Kurtosis: {kurtosis(df.SalePrice)}')
ax = sns.histplot(data=df, x='SalePrice', kde=True)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=45)

print(f'Kurtosis: {kurtosis(np.log(df.SalePrice))}')
sns.histplot(data=df, x=np.log(df['SalePrice']), kde=True)
plt.xlabel('SalePrice (Log Scale)')


def null_table(data):
    null_list = []
    for i in data:
        if data[i].isnull().any():
            null_list.append(data[i].isnull().value_counts())
    return pd.DataFrame(pd.concat(null_list, axis=1).T)
df_null = null_table(df)
print(df_null.shape)
df_null
df['Electrical'] = df.Electrical.fillna(mode(df.Electrical))
df['MasVnrType'] = df.MasVnrType.fillna(mode(df.MasVnrType))
df['MasVnrArea'] = df.MasVnrArea.fillna(median(df.MasVnrArea))
df['GarageYrBlt'] = df.GarageYrBlt.fillna(0).astype(int)
df_test_null = null_table(df_test)
print(df_test_null.shape)
df_test_null
df_test['MSZoning'] = df_test.MSZoning.fillna(mode(pd.concat([df['MSZoning'], df_test['MSZoning']], axis=0)))
df_test['Utilities'] = df_test.Utilities.fillna(mode(pd.concat([df['Utilities'], df_test['Utilities']], axis=0)))
df_test['Exterior1st'] = df_test.Exterior1st.fillna(mode(pd.concat([df['Exterior1st'], df_test['Exterior1st']], axis=0)))
df_test['Exterior2nd'] = df_test.Exterior2nd.fillna(mode(pd.concat([df['Exterior2nd'], df_test['Exterior2nd']], axis=0)))
df_test['MasVnrType'] = df_test.MasVnrType.fillna(mode(pd.concat([df['MasVnrType'], df_test['MasVnrType']], axis=0)))
df_test['BsmtFullBath'] = df_test.BsmtFullBath.fillna(mode(pd.concat([df['BsmtFullBath'], df_test['BsmtFullBath']], axis=0)))
df_test['BsmtHalfBath'] = df_test.BsmtHalfBath.fillna(mode(pd.concat([df['BsmtHalfBath'], df_test['BsmtHalfBath']], axis=0)))
df_test['KitchenQual'] = df_test.KitchenQual.fillna(mode(pd.concat([df['KitchenQual'], df_test['KitchenQual']], axis=0)))
df_test['Functional'] = df_test.Functional.fillna(mode(pd.concat([df['Functional'], df_test['Functional']], axis=0)))
df_test['SaleType'] = df_test.SaleType.fillna(mode(pd.concat([df['SaleType'], df_test['SaleType']], axis=0)))
df_test['GarageCars'] = df_test.GarageCars.fillna(mode(pd.concat([df['GarageCars'], df_test['GarageCars']], axis=0)))
df_test['GarageArea'] = df_test.GarageArea.fillna(median(pd.concat([df['GarageArea'], df_test['GarageArea']], axis=0)))
df_test['MasVnrArea'] = df_test.MasVnrArea.fillna(median(pd.concat([df['MasVnrArea'], df_test['MasVnrArea']], axis=0)))
df_test['BsmtFinSF1'] = df_test.BsmtFinSF1.fillna(median(pd.concat([df['BsmtFinSF1'], df_test['BsmtFinSF1']], axis=0)))
df_test['BsmtFinSF2'] = df_test.BsmtFinSF2.fillna(median(pd.concat([df['BsmtFinSF2'], df_test['BsmtFinSF2']], axis=0)))
df_test['BsmtUnfSF'] = df_test.BsmtUnfSF.fillna(median(pd.concat([df['BsmtUnfSF'], df_test['BsmtUnfSF']], axis=0)))
df_test['TotalBsmtSF'] = df_test.TotalBsmtSF.fillna(median(pd.concat([df['TotalBsmtSF'], df_test['TotalBsmtSF']], axis=0)))
df_test['GarageYrBlt'] = df_test.GarageYrBlt.fillna(0).astype(int)
print(null_table(df).shape)
print(null_table(df_test).shape)
df[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']] = df[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']].fillna('NA')
df_test[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']] = df_test[['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']].fillna('NA')
df = df.drop(['LotFrontage'], axis=1)
df_test = df_test.drop(['LotFrontage'], axis=1)
df_continuous = df[['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']]
df_test_continuous = df_test[['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']]
print(df_continuous.shape)
df_continuous.head()
print(df_test_continuous.shape)
df_test_continuous.head()
df['PriceAbvMedian'] = df['SalePrice'].apply(lambda x: 0 if x < median(df.SalePrice) else 1)
plt.figure(figsize=(18, 40))
for (i, j) in enumerate(df_continuous.columns):
    plt.subplot(10, 2, i + 1)
    ax = sns.scatterplot(data=df, x=f'{j}', y='SalePrice', hue='PriceAbvMedian')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.title(f'Sale Price vs. {j}')
    (m, b) = np.polyfit(x=df[j], y=df.SalePrice, deg=1)
    plt.plot(df[j], m * df[j] + b, c='red')
    corr_matrix = np.corrcoef(df[j], df.SalePrice)
    corr_xy = corr_matrix[0, 1]
    r_squared = round(corr_xy ** 2, 4)
    plt.legend(labels=[f'R^2 = {r_squared}', 'Price Above Median', 'Price Below Median'], framealpha=0.5, loc=0)
plt.tight_layout()

plt.figure(figsize=(5, 10))
sns.heatmap(pd.DataFrame(pd.concat([df['SalePrice'], df_continuous], axis=1).corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False)), annot=True)


def get_kurtosis(data):
    df_kurt = pd.DataFrame()
    df_kurt['Variable'] = data.columns
    for i in data.columns:
        df_kurt['Kurtosis'] = [kurtosis(data[f'{i}']) for i in data.columns]
    return df_kurt.sort_values(by=['Kurtosis'], ascending=False)
get_kurtosis(df_continuous)
df_discrete = df[['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageYrBlt', 'MoSold', 'YrSold']]
df_test_discrete = df_test[['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageYrBlt', 'MoSold', 'YrSold']]
print(df_discrete.shape)
df_discrete.head()
print(df_test_discrete.shape)
df_test_discrete.head()
plt.figure(figsize=(30, 18))
for (i, j) in enumerate(df_discrete.columns):
    (count, bin_edges) = np.histogram(df[f'{j}'])
    plt.subplot(3, 5, i + 1)
    sns.histplot(data=df, x=f'{j}', bins=bin_edges, hue='PriceAbvMedian')
    plt.title(f'Count vs. {j}')
plt.tight_layout()

plt.figure(figsize=(5, 10))
sns.heatmap(pd.DataFrame(pd.concat([df['SalePrice'], df_discrete], axis=1).corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False)), annot=True)
get_kurtosis(df_discrete)
df_nominal = df[['Alley', 'GarageType', 'MiscFeature', 'MSZoning', 'MSSubClass', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'SaleType', 'SaleCondition']]
df_test_nominal = df_test[['Alley', 'GarageType', 'MiscFeature', 'MSZoning', 'MSSubClass', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'SaleType', 'SaleCondition']]
print(df_nominal.shape)
df_nominal.head()
print(df_test_nominal.shape)
df_test_nominal.head()
plt.figure(figsize=(15, 60))
for (i, j) in enumerate(df_nominal.columns):
    plt.subplot(12, 2, i + 1)
    ax = sns.boxplot(data=df, x=f'{j}', y=df.SalePrice)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.title(f'SalePrice vs. {j}')
plt.tight_layout()

encoder_nominal = OrdinalEncoder()
df_nominal_ord = pd.DataFrame(encoder_nominal.fit_transform(df_nominal))
df_nominal_ord.columns = ['Alley', 'GarageType', 'MiscFeature', 'MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'SaleType', 'SaleCondition']
encoder_nominal_test = OrdinalEncoder()
df_test_nominal_ord = pd.DataFrame(encoder_nominal_test.fit_transform(df_test_nominal))
df_test_nominal_ord.columns = ['Alley', 'GarageType', 'MiscFeature', 'MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'SaleType', 'SaleCondition']
print(df_nominal_ord.shape)
df_nominal_ord.head()
print(df_test_nominal_ord.shape)
df_test_nominal_ord.head()
plt.figure(figsize=(5, 10))
sns.heatmap(pd.DataFrame(pd.concat([df['SalePrice'], df_nominal_ord], axis=1).corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False)), annot=True)
get_kurtosis(df_nominal_ord)
df_ordinal = df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'PoolQC', 'Fence']]
df_test_ordinal = df_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'PoolQC', 'Fence']]
print(df_ordinal.shape)
df_ordinal.head()
print(df_test_ordinal.shape)
df_test_ordinal.head()
plt.figure(figsize=(15, 60))
for (i, j) in enumerate(df_ordinal.columns):
    plt.subplot(12, 2, i + 1)
    ax = sns.boxplot(data=df, x=f'{j}', y=df.SalePrice)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.title(f'SalePrice vs. {j}')
plt.tight_layout()

encoder = OrdinalEncoder()
df_ordinal = pd.DataFrame(encoder.fit_transform(df_ordinal))
df_ordinal.columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'PoolQC', 'Fence']
encoder_test = OrdinalEncoder()
df_test_ordinal = pd.DataFrame(encoder_test.fit_transform(df_test_ordinal))
df_test_ordinal.columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'PoolQC', 'Fence']
print(df_ordinal.shape)
df_ordinal.head()
print(df_test_ordinal.shape)
df_test_ordinal.head()
plt.figure(figsize=(5, 10))
sns.heatmap(pd.DataFrame(pd.concat([df['SalePrice'], df_ordinal], axis=1).corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False)), annot=True)
get_kurtosis(df_ordinal)
df_concat = pd.concat([df_continuous, df_discrete, df_ordinal, df_nominal_ord], axis=1).astype(float)
df_test_concat = pd.concat([df_test_continuous, df_test_discrete, df_test_ordinal, df_test_nominal_ord], axis=1).astype(float)
print(df_concat.shape)
df_concat.head()
print(df_test_concat.shape)
df_test_concat.head()

def get_vif(data):
    df_vars = pd.DataFrame(np.log1p(data))
    df_vars_const = sma.add_constant(df_vars)
    df_vif = pd.DataFrame()
    df_vif['Feature'] = df_vars_const.columns
    df_vif['VIF'] = [variance_inflation_factor(df_vars_const.values, i) for i in range(df_vars_const.shape[1])]
    return df_vif.sort_values(by=['VIF'], ascending=False).head(25)
get_vif(df_concat)
mask = np.triu(np.ones_like(df_concat.corr(), dtype=bool))
plt.figure(figsize=(20, 20))
sns.heatmap(df_concat.corr(), mask=mask, linewidths=1, linecolor='black')

df_concat = df_concat.drop(['WoodDeckSF', 'OpenPorchSF', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'MSZoning', 'GarageCars', 'GarageArea', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd'], axis=1)
df_test_concat = df_test_concat.drop(['WoodDeckSF', 'OpenPorchSF', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'MSZoning', 'GarageCars', 'GarageArea', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd'], axis=1)
get_vif(df_concat)
print(df_concat.shape)
df_concat.head()
print(df_test_concat.shape)
df_test_concat.head()
X_features = np.log1p(df_concat)
X_test_features = np.log1p(df_test_concat)
Y_labels = df['SalePrice'].astype(float)
print(Y_labels.shape)
Y_labels.head()

def evaluate_model(name, model, X_train, Y_train):
    (x_train, x_val, y_train, y_val) = train_test_split(X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED)