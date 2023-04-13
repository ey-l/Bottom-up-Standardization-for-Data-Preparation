import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
(_input1.shape, _input0.shape)

def get_cols_with_missing_values(DataFrame):
    missing_na_columns = DataFrame.isnull().sum()
    return missing_na_columns[missing_na_columns > 0]
feature_with_na = get_cols_with_missing_values(_input1)
print(feature_with_na)
feature_with_na = get_cols_with_missing_values(_input0)
print(feature_with_na)
sns.distplot(_input1.get('SalePrice'), kde=False)
numerical_df = _input1.select_dtypes(exclude=['object'])
numerical_df = numerical_df.drop(['Id'], axis=1)
for column in numerical_df:
    plt.figure(figsize=(16, 4))
    sns.set_theme(style='whitegrid')
    sns.boxplot(numerical_df[column])
feature_train_not_test = [col for col in _input1.columns if col not in _input0.columns and col != 'SalePrice']
print(feature_train_not_test)
feature_test_not_train = [col for col in _input0.columns if col not in _input1.columns]
print(feature_test_not_train)
df_merge = pd.concat([_input0.assign(ind='test'), _input1.assign(ind='train')])
df_merge.head()
df_merge.info()
feature_rating_Qual = [col for col in df_merge.columns if 'Qual' in col and df_merge[col].dtypes == 'object']
feature_rating_Cond = [col for col in df_merge.columns if 'Cond' in col and col not in ['Condition1', 'Condition2', 'SaleCondition'] and (df_merge[col].dtypes == 'object')]
feature_rating_Qu = [col for col in df_merge.columns if 'Qu' in col and df_merge[col].dtypes == 'object' and (col not in feature_rating_Qual)]
feature_rating_QC = [col for col in df_merge.columns if 'QC' in col and df_merge[col].dtypes == 'object']
cat_feature_with_rating = feature_rating_Qual + feature_rating_Cond + feature_rating_Qu + feature_rating_QC
for x in cat_feature_with_rating:
    print(x)
cat_feature_with_legit_na = ['Alley', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'Fence', 'MiscFeature']
df_merge[cat_feature_with_legit_na].head()
ordinal_cat_features = cat_feature_with_rating + cat_feature_with_legit_na
df_merge[ordinal_cat_features].head()
df_merge[ordinal_cat_features] = df_merge[ordinal_cat_features].fillna('Missing')
print(get_cols_with_missing_values(df_merge[ordinal_cat_features]))
categorical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes == 'object' and cname != 'ind']
remaining_cat_cols = [cname for cname in categorical_cols if cname not in ordinal_cat_features]
numerical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes != 'object' and cname != 'SalePrice']
df_merge[remaining_cat_cols].head()
for col in remaining_cat_cols:
    df_merge[col] = df_merge[col].fillna(df_merge[col].mode()[0])
print(get_cols_with_missing_values(df_merge[remaining_cat_cols]))
df_merge[numerical_cols] = df_merge[numerical_cols].fillna(df_merge[numerical_cols].mean())
print(get_cols_with_missing_values(df_merge[numerical_cols]))
categorical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes == 'object' and df_merge[cname].nunique() < 10]
numerical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes != 'object']
my_cols = numerical_cols + categorical_cols
df_merge_clean = df_merge[my_cols].copy()
print(get_cols_with_missing_values(df_merge_clean))
df_merge_clean.head()
df_merge_clean = df_merge_clean.drop('Id', axis=1, inplace=False)
df_merge_clean['GarageYrBlt'] = df_merge_clean['GarageYrBlt'].astype('int')
df_merge_clean['GarageYrBlt'] = df_merge_clean['YrSold'] - df_merge_clean['GarageYrBlt']
df_merge_clean['YearBuilt'] = df_merge_clean['YrSold'] - df_merge_clean['YearBuilt']
df_merge_clean['YearRemodAdd'] = df_merge_clean['YrSold'] - df_merge_clean['YearRemodAdd']
df_merge_clean = df_merge_clean.drop(['YrSold'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['MoSold'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['TotalBsmtSF'], axis=1, inplace=False)
df_merge_clean['BsmtFinSF'] = df_merge_clean['BsmtFinSF1'] + df_merge_clean['BsmtFinSF2']
df_merge_clean = df_merge_clean.drop(['BsmtFinSF1'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['BsmtFinSF2'], axis=1, inplace=False)
df_merge_clean['TotalFlrSF'] = df_merge_clean['1stFlrSF'] + df_merge_clean['2ndFlrSF']
df_merge_clean = df_merge_clean.drop(['1stFlrSF'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['2ndFlrSF'], axis=1, inplace=False)
df_merge_clean['Total_Bath'] = df_merge_clean['FullBath'] + 0.5 * df_merge_clean['HalfBath'] + df_merge_clean['BsmtFullBath'] + 0.5 * df_merge_clean['BsmtHalfBath']
df_merge_clean = df_merge_clean.drop(['FullBath'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['HalfBath'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['BsmtFullBath'], axis=1, inplace=False)
df_merge_clean = df_merge_clean.drop(['BsmtHalfBath'], axis=1, inplace=False)
import scipy.stats
numerical_cols = [cname for cname in df_merge_clean.columns if df_merge_clean[cname].dtypes != 'object' and cname != 'SalePrice']
skew_df = pd.DataFrame(numerical_cols, columns=['Feature'])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(df_merge_clean[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)
skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)
skew_df
df_merge_clean[numerical_cols].describe()
for column in skew_df.query('Skewed == True')['Feature'].values:
    df_merge_clean[column] = np.log1p(df_merge_clean[column])
df_merge_clean[cat_feature_with_rating]
for col in cat_feature_with_rating:
    if 'Missing' in df_merge_clean[col].value_counts().index:
        df_merge_clean[col] = df_merge_clean[col].map({'Missing': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    else:
        df_merge_clean[col] = df_merge_clean[col].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df_merge_clean[cat_feature_with_rating]
df_merge_clean[cat_feature_with_legit_na]
df_merge_clean['BsmtExposure'] = df_merge_clean['BsmtExposure'].map({'Missing': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).astype('int')
df_merge_clean['BsmtFinType1'] = df_merge_clean['BsmtFinType1'].map({'Missing': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype('int')
df_merge_clean['BsmtFinType2'] = df_merge_clean['BsmtFinType2'].map({'Missing': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).astype('int')
df_merge_clean['GarageFinish'] = df_merge_clean['GarageFinish'].map({'Missing': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype('int')
df_merge_clean['Fence'] = df_merge_clean['Fence'].map({'Missing': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).astype('int')
df_merge_clean['LotShape'] = df_merge_clean['LotShape'].map({'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}).astype('int')
df_merge_clean['LandContour'] = df_merge_clean['LandContour'].map({'Low': 1, 'Bnk': 2, 'HLS': 3, 'Lvl': 4}).astype('int')
df_merge_clean['Utilities'] = df_merge_clean['Utilities'].map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}).astype('int')
df_merge_clean['LandSlope'] = df_merge_clean['LandSlope'].map({'Sev': 1, 'Mod': 2, 'Gtl': 3}).astype('int')
df_merge_clean['CentralAir'] = df_merge_clean['CentralAir'].map({'N': 0, 'Y': 1}).astype('int')
df_merge_clean['PavedDrive'] = df_merge_clean['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2}).astype('int')
cat_remaining_to_encode = [col for col in df_merge_clean.columns if df_merge_clean[col].dtypes == 'object' and col != 'ind']
print(cat_remaining_to_encode)
df_merge_clean_dummies = pd.get_dummies(df_merge_clean[cat_remaining_to_encode], drop_first=True)
df_merge_clean = df_merge_clean.drop(cat_remaining_to_encode, axis=1, inplace=False)
df_merge_clean = pd.concat([df_merge_clean, df_merge_clean_dummies], axis=1)