import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
pd.set_option('display.max_columns', None)
train_dataset.head()
train_dataset.shape
print('Dataset Info')
print(train_dataset.info())
print('---------------------------------------------------------------')
print('')
print('Dataset Stats')
print(train_dataset.describe())
print('---------------------------------------------------------------')
print('Dataset null values:')
pd.set_option('display.max_rows', None)
pd.DataFrame(train_dataset.isna().any()).reset_index()
test_dataset_with_out = test_dataset.copy()
test_dataset_with_out['SalePrice'] = 1
complete_dataset = pd.concat([train_dataset, test_dataset_with_out])
complete_dataset.shape
numerical_columns = train_dataset.describe().columns.tolist()
categorical_columns = train_dataset.columns[list((i not in numerical_columns for i in train_dataset.columns.values))].tolist()
numerical_columns.remove('Id')
numerical_columns.remove('MSSubClass')
categorical_columns.append('MSSubClass')
sns.set(rc={'figure.figsize': (5, 4)})
for i in categorical_columns:

    plt.xlabel(i)

for i in categorical_columns:
    sns.boxplot(x=i, y='SalePrice', data=train_dataset)

sns.set(rc={'figure.figsize': (8, 6)})
for i in numerical_columns:
    sns.boxplot(y=i, data=train_dataset)

for i in numerical_columns:
    sns.histplot(x=i, data=train_dataset)

data_skew = pd.DataFrame(train_dataset[numerical_columns].skew().sort_values(ascending=False))
data_skew
sns.set(rc={'figure.figsize': (30, 12)})
sns.heatmap(train_dataset.corr(), annot=True, cmap='coolwarm')
rel_list = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
for i in rel_list:
    sns.jointplot(x=i, y='SalePrice', data=train_dataset, kind='reg')
    plt.title(i + ' vs SalePrice')

complete_dataset_1 = complete_dataset.copy()
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_dataset.duplicated().any()
complete_dataset.loc[:, ['LotFrontage']] = median_imputer.fit_transform(complete_dataset.loc[:, ['LotFrontage']])
complete_dataset.loc[:, ['BsmtFinSF1']] = median_imputer.fit_transform(complete_dataset.loc[:, ['BsmtFinSF1']])
complete_dataset.loc[:, ['BsmtFinSF2']] = median_imputer.fit_transform(complete_dataset.loc[:, ['BsmtFinSF2']])
complete_dataset.loc[:, ['BsmtUnfSF']] = median_imputer.fit_transform(complete_dataset.loc[:, ['BsmtUnfSF']])
complete_dataset.loc[:, ['TotalBsmtSF']] = median_imputer.fit_transform(complete_dataset.loc[:, ['TotalBsmtSF']])
complete_dataset.loc[:, ['BsmtFullBath']] = median_imputer.fit_transform(complete_dataset.loc[:, ['BsmtFullBath']])
complete_dataset.loc[:, ['BsmtHalfBath']] = median_imputer.fit_transform(complete_dataset.loc[:, ['BsmtHalfBath']])
complete_dataset.loc[:, ['GarageCars']] = median_imputer.fit_transform(complete_dataset.loc[:, ['GarageCars']])
complete_dataset.loc[:, ['GarageArea']] = median_imputer.fit_transform(complete_dataset.loc[:, ['GarageArea']])
complete_dataset.loc[:, ['MasVnrArea']] = median_imputer.fit_transform(complete_dataset.loc[:, ['MasVnrArea']])
complete_dataset.GarageYrBlt.replace(np.nan, 0, inplace=True)
complete_dataset.Alley.replace(np.nan, 'nil', inplace=True)
complete_dataset.BsmtQual.replace(np.nan, 'nil', inplace=True)
complete_dataset.BsmtFinType1.replace(np.nan, 'nil', inplace=True)
complete_dataset.FireplaceQu.replace(np.nan, 'nil', inplace=True)
complete_dataset.GarageType.replace(np.nan, 'nil', inplace=True)
complete_dataset.GarageFinish.replace(np.nan, 'nil', inplace=True)
complete_dataset.PoolQC.replace(np.nan, 'nil', inplace=True)
complete_dataset.loc[:, ['MasVnrType']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['MasVnrType']])
complete_dataset.loc[:, ['BsmtCond']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['BsmtCond']])
complete_dataset.loc[:, ['BsmtExposure']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['BsmtExposure']])
complete_dataset.loc[:, ['BsmtFinType2']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['BsmtFinType2']])
complete_dataset.loc[:, ['Electrical']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['Electrical']])
complete_dataset.loc[:, ['GarageQual']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['GarageQual']])
complete_dataset.loc[:, ['GarageCond']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['GarageCond']])
complete_dataset.loc[:, ['Fence']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['Fence']])
complete_dataset.loc[:, ['MiscFeature']] = mode_imputer.fit_transform(complete_dataset.loc[:, ['MiscFeature']])
encoded_complete_dataset = pd.get_dummies(complete_dataset.iloc[:, :-1], columns=categorical_columns)
encoded_complete_dataset['SalePrice'] = complete_dataset['SalePrice']
skewed_features = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for i in skewed_features:
    encoded_complete_dataset[i] = np.log(encoded_complete_dataset[i].values)
encoded_complete_dataset.head()
train_dataset_cleaned = encoded_complete_dataset.loc[encoded_complete_dataset.Id <= 1460, :]
test_dataset_cleaned = encoded_complete_dataset.loc[encoded_complete_dataset.Id > 1460, :]
test_dataset_cleaned = test_dataset_cleaned.drop('SalePrice', axis=1)
X_train = train_dataset_cleaned.iloc[:, :-1]
y_train = train_dataset_cleaned.iloc[:, -1]
X_test = test_dataset_cleaned.copy()
X_test = X_test.iloc[:, :-1]
from sklearn.ensemble import ExtraTreesClassifier
y_train_pca = y_train.copy()
y_train_pca = y_train_pca.astype('int')
model = ExtraTreesClassifier(n_estimators=10, random_state=42)