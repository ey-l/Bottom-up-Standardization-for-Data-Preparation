import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
num_null_cols = [i for i in _input1.columns if _input1[i].isnull().any()]
print(f'Number of features have null values: {len(num_null_cols)}')
print()
print('Each feature with the corresponding null value count')
_input1.isnull().sum()
Target = 'SalePrice'
a = [np.nan, None, [], {}, 'NaN', 'Null', 'NULL', 'None', 'NA', '?', '-', '.', '', ' ', '   ']
for c in _input1.columns:
    string_null = np.array([x in a[2:] for x in _input1[c]])
    print(c, _input1[c].isnull().sum(), string_null.sum())
_input1.hist(figsize=(20, 20))
_input1.info()
percent_missing = _input1.isnull().sum() * 100 / len(_input1)
missing_value_df = pd.DataFrame({'column_name': _input1.columns, 'percent_missing': percent_missing})
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
_input1 = _input1.drop(features_to_throw, axis=1, inplace=False)
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input1.head()
print(f'Num of features lift: {_input1.shape[1]}')
MasVnrArea_median = _input1['MasVnrArea'].median()
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(MasVnrArea_median)
MasVnrType_mode = _input1['MasVnrType'].mode()[0]
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(MasVnrType_mode)
BsmtQual_mode = _input1['BsmtQual'].mode()[0]
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(BsmtQual_mode)
BsmtCond_mode = _input1['BsmtCond'].mode()[0]
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(BsmtCond_mode)
BsmtExposure_mode = _input1['BsmtExposure'].mode()[0]
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(BsmtExposure_mode)
BsmtFinType1_mode = _input1['BsmtFinType1'].mode()[0]
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(BsmtFinType1_mode)
BsmtFinType2_mode = _input1['BsmtFinType2'].mode()[0]
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(BsmtFinType2_mode)
Electrical_mode = _input1['Electrical'].mode()[0]
_input1['Electrical'] = _input1['Electrical'].fillna(Electrical_mode)
_input1.head()
print(f'Num of features lift: {_input1.shape[1]}')
print(_input1.isnull().sum())
print('\nAwesome!!')
percent_missing_test = _input0.isnull().sum() * 100 / len(_input0)
missing_value_df_test = pd.DataFrame({'column_name': _input0.columns, 'percent_missing': percent_missing_test})
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
_input0 = _input0.drop(features_to_throw_test, axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
print(f'Num of features lift: {_input0.shape[1]}')
for c in features_to_impute_test:
    _input0[c].hist()
    plt.title(c)
for c in features_to_impute_test:
    print(c, len(_input0[c].unique()), _input0[c].dtype)
MasVnrArea_median_test = _input0['MasVnrArea'].median()
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(MasVnrArea_median_test)
MasVnrType_mode_test = _input0['MasVnrType'].mode()[0]
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(MasVnrType_mode_test)
MSZoning_mode_test = _input0['MSZoning'].mode()[0]
_input0['MSZoning'] = _input0['MSZoning'].fillna(MSZoning_mode_test)
Utilities_mode_test = _input0['Utilities'].mode()[0]
_input0['Utilities'] = _input0['Utilities'].fillna(Utilities_mode_test)
Exterior1st_mode_test = _input0['Exterior1st'].mode()[0]
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(Exterior1st_mode_test)
Exterior2nd_mode_test = _input0['Exterior2nd'].mode()[0]
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(Exterior2nd_mode_test)
BsmtFinSF1_median_test = _input0['BsmtFinSF1'].median()
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(BsmtFinSF1_median_test)
BsmtFinSF2_median_test = _input0['BsmtFinSF2'].median()
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(BsmtFinSF2_median_test)
BsmtUnfSF_median_test = _input0['BsmtUnfSF'].median()
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(BsmtUnfSF_median_test)
TotalBsmtSF_median_test = _input0['TotalBsmtSF'].median()
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(TotalBsmtSF_median_test)
TotalBsmtSF_median_test = _input0['TotalBsmtSF'].median()
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(TotalBsmtSF_median_test)
BsmtFullBath_mode_test = _input0['BsmtFullBath'].mode()[0]
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(BsmtFullBath_mode_test)
BsmtHalfBath_mode_test = _input0['BsmtHalfBath'].mode()[0]
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(BsmtHalfBath_mode_test)
KitchenQual_mode_test = _input0['KitchenQual'].mode()[0]
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(KitchenQual_mode_test)
Functional_mode_test = _input0['Functional'].mode()[0]
_input0['Functional'] = _input0['Functional'].fillna(Functional_mode_test)
GarageCars_mode_test = _input0['GarageCars'].mode()[0]
_input0['GarageCars'] = _input0['GarageCars'].fillna(GarageCars_mode_test)
GarageArea_mean_test = _input0['GarageArea'].mean()
_input0['GarageArea'] = _input0['GarageArea'].fillna(GarageArea_mean_test)
SaleType_mode_test = _input0['SaleType'].mode()[0]
_input0['SaleType'] = _input0['SaleType'].fillna(SaleType_mode_test)
BsmtQual_mode_test = _input0['BsmtQual'].mode()[0]
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(BsmtQual_mode_test)
BsmtCond_mode_test = _input0['BsmtCond'].mode()[0]
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(BsmtCond_mode_test)
BsmtExposure_mode_test = _input0['BsmtExposure'].mode()[0]
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(BsmtExposure_mode_test)
BsmtFinType1_mode_test = _input0['BsmtFinType1'].mode()[0]
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(BsmtFinType1_mode_test)
BsmtFinType2_mode_test = _input0['BsmtFinType2'].mode()[0]
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(BsmtFinType2_mode_test)
_input0.head()
print(f'Num of features lift: {_input0.shape[1]}')
print(_input0.isnull().sum())
print('\nAwesome!!!!')
cat_lst = []
for c in _input0:
    if _input0[c].dtype == 'object':
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
        final_df = final_df.drop([fields], axis=1, inplace=False)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
Main_Train = _input1.copy()
print(f'Train Data Shape : {_input1.shape}', f'\nTest Data Shape  : {_input0.shape}')
final_df = pd.concat([_input1, _input0], axis=0)
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
X_Test_df = X_Test_df.drop(['SalePrice'], inplace=False, axis=1)
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