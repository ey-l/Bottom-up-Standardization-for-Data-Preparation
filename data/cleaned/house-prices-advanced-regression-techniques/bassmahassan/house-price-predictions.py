import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn import feature_selection
import warnings
warnings.filterwarnings('ignore')
SEED = 42
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()
test_data.head()
train_data.shape
test_data.shape
train_data.info()
dfs = [train_data, test_data]
for df in dfs:
    temp = df.isnull().sum()
    print(temp.loc[temp != 0], '\n')
test_data.info()
train_data['LT_Salesprice'] = np.log1p(train_data['SalePrice'])
plt.hist(train_data['LT_Salesprice'], color='black')

train_data['LT_Salesprice'].skew()
plt.figure(figsize=(20, 20))
sns.heatmap(train_data.corr())


def data_cleaning(df):
    df['MSZoning'].fillna(value=df['MSZoning'].mode()[0], inplace=True)
    df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'MiscFeature', 'Fence'], axis='columns', inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].dropna().mean(), inplace=True)
    for Bsmt in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']:
        df[Bsmt].fillna(df[Bsmt].mode()[0], inplace=True)
    for garage in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars', 'GarageArea']:
        df[garage].fillna(df[garage].mode()[0], inplace=True)
    for other in ['SaleType', 'Functional', 'KitchenQual', 'Electrical', 'MasVnrType', 'Exterior1st', 'Exterior2nd', 'Utilities', 'MasVnrArea']:
        df[other].fillna(df[other].mode()[0], inplace=True)
    numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    print('Number of numerical variables: ', len(numerical_features))
    df[numerical_features].head()
    year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
    year_feature
    df.groupby('YrSold')['SalePrice'].median().plot()

    discrete_feature = [feature for feature in numerical_features if len(df[feature].unique()) < 25 and feature not in year_feature + ['id']]
    continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
    for feature in continuous_feature:
        data = df.copy()
        if 0 in data[feature].unique():
            pass
        else:
            data[feature] = np.log(data[feature])
            data['SalePrice'] = np.log(data['SalePrice'])
            plt.scatter(data[feature], data['SalePrice'])
            plt.xlabel(feature)
            plt.ylabel('Salesprice')

    for feature in continuous_feature:
        data = df.copy()
        if 0 in data[feature].unique():
            pass
        else:
            data[feature] = np.log(data[feature])
            data.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)

    for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
        df[feature] = df['YrSold'] - df[feature]
    categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
    len(categorical_features)
    for feature in categorical_features:
        temp = df.groupby(feature)['SalePrice'].count() / len(df)
        temp_df = temp[temp > 0.01].index
        df[feature] = np.where(df[feature].isin(temp_df), df[feature], 'Rare_var')
    df.shape
    for features in categorical_features:
        dummies = pd.get_dummies(df[features])
        merged = pd.concat([df, dummies], axis='columns')
        df = merged.copy()
    for feature in categorical_features:
        df.drop(feature, axis='columns', inplace=True)
    df.drop('LT_Salesprice', axis='columns', inplace=True)
    return df
Dataset = pd.concat([train_data, test_data])
clean_data = data_cleaning(Dataset)
clean_test = clean_data.iloc[1460:, :]

clean_train = clean_data.iloc[:1460, :]

X_train = clean_train.drop('SalePrice', axis='columns')
y_train = clean_train.SalePrice
X_test = clean_test.drop('SalePrice', axis='columns')
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=25)