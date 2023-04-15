import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
X_Train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
X_Test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
X_Train.head()
num_null_cols = [i for i in X_Train.columns if X_Train[i].isnull().any()]
print(f'Number of features have null values: {len(num_null_cols)}')
print()
print('Each feature with the corresponding null value count')
X_Train.isnull().sum()
Target = 'SalePrice'
a = [np.nan, None, [], {}, 'NaN', 'Null', 'NULL', 'None', 'NA', '?', '-', '.', '', ' ', '   ']
for c in X_Train.columns:
    string_null = np.array([x in a[2:] for x in X_Train[c]])
    print(c, X_Train[c].isnull().sum(), string_null.sum())
X_Train.hist(figsize=(20, 20))
X_Train.info()
percent_missing = X_Train.isnull().sum() * 100 / len(X_Train)
missing_value_df = pd.DataFrame({'column_name': X_Train.columns, 'percent_missing': percent_missing})
impute_lst = []
throw_lst = []
for i in range(0, len(missing_value_df['percent_missing'])):
    if missing_value_df['percent_missing'][i] <= 5 and missing_value_df['percent_missing'][i] > 0:
        impute_lst.append(missing_value_df['column_name'][i])
    elif missing_value_df['percent_missing'][i] > 5:
        throw_lst.append(missing_value_df['column_name'][i])
features_to_impute = impute_lst
features_to_throw = throw_lst
print('Features to Impute: ')
print(len(features_to_impute), features_to_impute)
print()
print('Features to Throw: ')
print(len(features_to_throw), features_to_throw)
X_Train.drop(features_to_throw, axis=1, inplace=True)
X_Train.drop('Id', axis=1, inplace=True)
X_Train.head()
print(f'Num of features lift: {X_Train.shape[1]}')
MasVnrArea_median = X_Train['MasVnrArea'].median()
X_Train['MasVnrArea'] = X_Train['MasVnrArea'].fillna(MasVnrArea_median)
MasVnrType_mode = X_Train['MasVnrType'].mode()[0]
X_Train['MasVnrType'] = X_Train['MasVnrType'].fillna(MasVnrType_mode)
BsmtQual_mode = X_Train['BsmtQual'].mode()[0]
X_Train['BsmtQual'] = X_Train['BsmtQual'].fillna(BsmtQual_mode)
BsmtCond_mode = X_Train['BsmtCond'].mode()[0]
X_Train['BsmtCond'] = X_Train['BsmtCond'].fillna(BsmtCond_mode)
BsmtExposure_mode = X_Train['BsmtExposure'].mode()[0]
X_Train['BsmtExposure'] = X_Train['BsmtExposure'].fillna(BsmtExposure_mode)
BsmtFinType1_mode = X_Train['BsmtFinType1'].mode()[0]
X_Train['BsmtFinType1'] = X_Train['BsmtFinType1'].fillna(BsmtFinType1_mode)
BsmtFinType2_mode = X_Train['BsmtFinType2'].mode()[0]
X_Train['BsmtFinType2'] = X_Train['BsmtFinType2'].fillna(BsmtFinType2_mode)
Electrical_mode = X_Train['Electrical'].mode()[0]
X_Train['Electrical'] = X_Train['Electrical'].fillna(Electrical_mode)
X_Train.head()
print(f'Num of features lift: {X_Train.shape[1]}')
print(X_Train.isnull().sum())
print('\nAwesome!!')
percent_missing_test = X_Test.isnull().sum() * 100 / len(X_Test)
missing_value_df_test = pd.DataFrame({'column_name': X_Test.columns, 'percent_missing': percent_missing_test})
impute_lst_2 = []
throw_lst_2 = []
for i in range(0, len(missing_value_df_test['percent_missing'])):
    if missing_value_df_test['percent_missing'][i] <= 5 and missing_value_df_test['percent_missing'][i] > 0:
        impute_lst_2.append(missing_value_df_test['column_name'][i])
    elif missing_value_df_test['percent_missing'][i] > 5:
        throw_lst_2.append(missing_value_df_test['column_name'][i])
features_to_impute_test = impute_lst_2
features_to_throw_test = throw_lst_2
print('Features to Impute: ')
print(len(features_to_impute_test), features_to_impute_test)
print()
print('Features to Throw: ')
print(len(features_to_throw_test), features_to_throw_test)
X_Test.drop(features_to_throw_test, axis=1, inplace=True)
X_Test.drop('Id', axis=1, inplace=True)
print(f'Num of features lift: {X_Test.shape[1]}')
for c in features_to_impute_test:
    X_Test[c].hist()
    plt.title(c)

for c in features_to_impute_test:
    print(c, len(X_Test[c].unique()), X_Test[c].dtype)
MasVnrArea_median_test = X_Test['MasVnrArea'].median()
X_Test['MasVnrArea'] = X_Test['MasVnrArea'].fillna(MasVnrArea_median_test)
MasVnrType_mode_test = X_Test['MasVnrType'].mode()[0]
X_Test['MasVnrType'] = X_Test['MasVnrType'].fillna(MasVnrType_mode_test)
MSZoning_mode_test = X_Test['MSZoning'].mode()[0]
X_Test['MSZoning'] = X_Test['MSZoning'].fillna(MSZoning_mode_test)
Utilities_mode_test = X_Test['Utilities'].mode()[0]
X_Test['Utilities'] = X_Test['Utilities'].fillna(Utilities_mode_test)
Exterior1st_mode_test = X_Test['Exterior1st'].mode()[0]
X_Test['Exterior1st'] = X_Test['Exterior1st'].fillna(Exterior1st_mode_test)
Exterior2nd_mode_test = X_Test['Exterior2nd'].mode()[0]
X_Test['Exterior2nd'] = X_Test['Exterior2nd'].fillna(Exterior2nd_mode_test)
BsmtFinSF1_median_test = X_Test['BsmtFinSF1'].median()
X_Test['BsmtFinSF1'] = X_Test['BsmtFinSF1'].fillna(BsmtFinSF1_median_test)
BsmtFinSF2_median_test = X_Test['BsmtFinSF2'].median()
X_Test['BsmtFinSF2'] = X_Test['BsmtFinSF2'].fillna(BsmtFinSF2_median_test)
BsmtUnfSF_median_test = X_Test['BsmtUnfSF'].median()
X_Test['BsmtUnfSF'] = X_Test['BsmtUnfSF'].fillna(BsmtUnfSF_median_test)
TotalBsmtSF_median_test = X_Test['TotalBsmtSF'].median()
X_Test['TotalBsmtSF'] = X_Test['TotalBsmtSF'].fillna(TotalBsmtSF_median_test)
TotalBsmtSF_median_test = X_Test['TotalBsmtSF'].median()
X_Test['TotalBsmtSF'] = X_Test['TotalBsmtSF'].fillna(TotalBsmtSF_median_test)
BsmtFullBath_mode_test = X_Test['BsmtFullBath'].mode()[0]
X_Test['BsmtFullBath'] = X_Test['BsmtFullBath'].fillna(BsmtFullBath_mode_test)
BsmtHalfBath_mode_test = X_Test['BsmtHalfBath'].mode()[0]
X_Test['BsmtHalfBath'] = X_Test['BsmtHalfBath'].fillna(BsmtHalfBath_mode_test)
KitchenQual_mode_test = X_Test['KitchenQual'].mode()[0]
X_Test['KitchenQual'] = X_Test['KitchenQual'].fillna(KitchenQual_mode_test)
Functional_mode_test = X_Test['Functional'].mode()[0]
X_Test['Functional'] = X_Test['Functional'].fillna(Functional_mode_test)
GarageCars_mode_test = X_Test['GarageCars'].mode()[0]
X_Test['GarageCars'] = X_Test['GarageCars'].fillna(GarageCars_mode_test)
GarageArea_mean_test = X_Test['GarageArea'].mean()
X_Test['GarageArea'] = X_Test['GarageArea'].fillna(GarageArea_mean_test)
SaleType_mode_test = X_Test['SaleType'].mode()[0]
X_Test['SaleType'] = X_Test['SaleType'].fillna(SaleType_mode_test)
BsmtQual_mode_test = X_Test['BsmtQual'].mode()[0]
X_Test['BsmtQual'] = X_Test['BsmtQual'].fillna(BsmtQual_mode_test)
BsmtCond_mode_test = X_Test['BsmtCond'].mode()[0]
X_Test['BsmtCond'] = X_Test['BsmtCond'].fillna(BsmtCond_mode_test)
BsmtExposure_mode_test = X_Test['BsmtExposure'].mode()[0]
X_Test['BsmtExposure'] = X_Test['BsmtExposure'].fillna(BsmtExposure_mode_test)
BsmtFinType1_mode_test = X_Test['BsmtFinType1'].mode()[0]
X_Test['BsmtFinType1'] = X_Test['BsmtFinType1'].fillna(BsmtFinType1_mode_test)
BsmtFinType2_mode_test = X_Test['BsmtFinType2'].mode()[0]
X_Test['BsmtFinType2'] = X_Test['BsmtFinType2'].fillna(BsmtFinType2_mode_test)
X_Test.head()
print(f'Num of features lift: {X_Test.shape[1]}')
print(X_Test.isnull().sum())
print('\nAwesome!!!!')
cat_lst = []
for c in X_Test:
    if X_Test[c].dtype == 'object':
        cat_lst.append(c)
print(cat_lst)
print()
print(f'Num of Categorical Features to encode: {len(cat_lst)}')

def onehot_encoding_categorical_concat(multcolumns):
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
Main_Train = X_Train.copy()
print(f'Train Data Shape : {X_Train.shape}', f'\nTest Data Shape  : {X_Test.shape}')
final_df = pd.concat([X_Train, X_Test], axis=0)
final_df.head()
final_df.shape
final_df = onehot_encoding_categorical_concat(cat_lst)
print(final_df.shape)
print("\nNow we one hot encoded the entire dataframe 'train + test' 😉")
final_df.head()
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
print(final_df.shape)
final_df.head(10)
X_Train_df = final_df.iloc[:1460, :]
X_Test_df = final_df.iloc[1460:, :]
print('Before Drop: ')
print(X_Test_df.shape)
X_Test_df.drop(['SalePrice'], inplace=True, axis=1)
print('After Drop: ')
X_Test_df.shape
x = X_Train_df.drop(['SalePrice'], axis=1)
y = X_Train_df['SalePrice']
x.head(10)
import xgboost
regressor = xgboost.XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV
max_depth = [2, 3, 5, 10, 15]
learning_rate = [0.1, 0.2]
hyperparameters = {'max_depth': max_depth, 'learning_rate': learning_rate}
random_search_cv = RandomizedSearchCV(estimator=regressor, param_distributions=hyperparameters, cv=5, n_iter=100, scoring='neg_mean_absolute_error', n_jobs=4, verbose=5, return_train_score=True, random_state=42)