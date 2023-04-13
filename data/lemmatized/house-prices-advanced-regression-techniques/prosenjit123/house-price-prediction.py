import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 100
sns.set(style='whitegrid')
from collections import Counter
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Shape of train and test : ', _input1.shape, _input0.shape)
_input1.head(10)
_input1.isnull().sum() / len(_input1)
df = _input1.nunique().sort_values().reset_index()
df.columns = ['Features', 'UniqueCount']
cols = df[df['UniqueCount'] <= 10]['Features'].values
print('Set of columns having 10 or less distinct classes :\n\n', cols)
print('\nPercentage of each class  :\n')
for features in cols:
    (c, c_percentage) = (dict(_input1[features].value_counts()), dict(_input1[features].value_counts(normalize=True)))
    print(features, c, c_percentage)

def fn_feature_engg(df):
    cols_to_del = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities', 'CentralAir', 'Street', 'BsmtHalfBath', 'LandSlope', 'PavedDrive', 'BsmtCond', 'KitchenAbvGr', 'Electrical', 'GarageQual', 'GarageCond', 'Heating', 'RoofMatl', 'Condition2', 'PoolArea']
    df = df.drop(cols_to_del, axis=1)
    fill_dict = {('MasVnrType',): 'None', ('MasVnrArea', 'GarageYrBlt', 'LotFrontage'): 0, ('BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish'): 'NA'}
    for (keys, values) in fill_dict.items():
        for cols in keys:
            df[cols] = df[cols].fillna(values, inplace=False)
    cols_to_impute_median = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']
    cols_to_impute_mode = ['Exterior1st', 'Exterior2nd', 'BsmtFullBath', 'SaleType', 'GarageCars', 'MSZoning', 'KitchenQual', 'Functional']
    for cols in cols_to_impute_median:
        df[cols] = df[cols].fillna(df[cols].median())
    for cols in cols_to_impute_mode:
        df[cols] = df[cols].fillna(df[cols].mode()[0])
    cat_cols_1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu']
    cat_1 = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
    cat_1_mapping = {'Ex': 6, 'Gd': 5, 'TA': 4, 'Fa': 3, 'Po': 2, 'NA': 1}
    cat_cols_2 = ['BsmtFinType1', 'BsmtFinType2']
    cat_2 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
    cat_2_mapping = {'GLQ': 7, 'ALQ': 6, 'BLQ': 5, 'Rec': 4, 'LwQ': 3, 'Unf': 2, 'NA': 1}
    cat_cols_3 = ['BsmtExposure']
    cat_3 = ['Gd', 'Av', 'Mn', 'No', 'NA']
    cat_3_mapping = {'Gd': 5, 'Av': 4, 'Mn': 3, 'No': 2, 'NA': 1}
    cat_cols_4 = ['LotShape']
    cat_4 = ['Reg', 'IR1', 'IR2', 'IR3']
    cat_4_mapping = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
    cat_cols_5 = ['GarageFinish']
    cat_5 = ['Fin', 'RFn', 'Unf', 'NA']
    cat_5_mapping = {'Fin': 4, 'RFn': 3, 'Unf': 2, 'NA': 1}
    for cols in cat_cols_1:
        df[cols] = df[cols].map(cat_1_mapping)
    for cols in cat_cols_2:
        df[cols] = df[cols].map(cat_2_mapping)
    for cols in cat_cols_3:
        df[cols] = df[cols].map(cat_3_mapping)
    for cols in cat_cols_4:
        df[cols] = df[cols].map(cat_4_mapping)
    for cols in cat_cols_5:
        df[cols] = df[cols].map(cat_5_mapping)
    df['YearsOld'] = df['YrSold'] - df['YearBuilt']
    df['IsRemod'] = (df['YearBuilt'] < df['YearRemodAdd']).astype(int)
    df['YearsRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['YearsOldGarage'] = df['YrSold'] - df['GarageYrBlt']
    df[df['YearsOld'] < 0]['YearsOld'] = 0
    df[df['YearsRemod'] < 0]['YearsRemod'] = 0
    df[df['YearsOldGarage'] < 0]['YearsOldGarage'] = 0
    cols_to_del = ['YrSold', 'YearBuilt', 'MoSold', 'YearRemodAdd', 'GarageYrBlt']
    df = df.drop(cols_to_del, axis=1)
    print('Shape of the return dataframe : ', df.shape)
    return df
train = fn_feature_engg(_input1)
train.head()
train.isnull().sum() / len(train)
target = 'SalePrice'
from scipy import stats
y = train[target]
(fig, ax) = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
plt.subplot(1, 2, 1)
sns.distplot(y)
plt.subplot(1, 2, 2)
stats.probplot(y, plot=plt)
plt.tight_layout()
y = np.log1p(train[target])
(fig, ax) = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
plt.subplot(1, 2, 1)
sns.distplot(y)
plt.subplot(1, 2, 2)
stats.probplot(y, plot=plt)
plt.tight_layout()
train[target] = np.log1p(train[target])
train.info()
categorical_columns = list(train.select_dtypes(include=['object']).columns)
categorical_columns = categorical_columns + ['MSSubClass', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'IsRemod']
Categorical_ordinal_columns = ['LotShape', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish']
Categorical_nominal_columns = [i for i in categorical_columns if i not in Categorical_ordinal_columns]
numerical_cols = [i for i in train.columns if i not in categorical_columns]
print('Number of categorical columns :', len(categorical_columns))
print(categorical_columns)
print('\nNumber of categorical ordinal columns :', len(Categorical_ordinal_columns))
print(Categorical_ordinal_columns)
print('\nNumber of categorical nominal columns :', len(Categorical_nominal_columns))
print(Categorical_nominal_columns)
print('\nNumber of numerical columns :', len(numerical_cols))
print(numerical_cols)
train[numerical_cols].corr()[target].abs().sort_values(ascending=False)
threshold = 0.2
df = train[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False).reset_index()
df.columns = ['Features', 'Correlation']
numerical_cols = df[df['Correlation'] >= threshold]['Features'].to_list()
print(numerical_cols, target)
(fig, ax) = plt.subplots(len(Categorical_ordinal_columns) // 5 + 1, 5)
fig.set_size_inches(20, 10)
for (idx, col) in enumerate(Categorical_ordinal_columns):
    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[idx // 5, idx % 5])
plt.tight_layout()
(fig, ax) = plt.subplots(len(Categorical_nominal_columns) // 5 + 1, 5)
fig.set_size_inches(30, 20)
for (idx, col) in enumerate(Categorical_nominal_columns):
    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[idx // 5, idx % 5])
plt.tight_layout()
selected_columns = numerical_cols + categorical_columns
print('Number of columns selected {0} \n\n{1}'.format(len(selected_columns), selected_columns))
print('\nTarget column : ', target)
feature_train = train[selected_columns]
(fig, ax) = plt.subplots(figsize=(30, 10))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(feature_train.corr().abs(), annot=True, fmt='.2f', cmap=cmap, ax=ax)
cols_to_remove = [target]
selected_columns = [i for i in selected_columns if i not in cols_to_remove]
print('Number of columns selected {0} \n\n{1}'.format(len(selected_columns), selected_columns))
y_train = feature_train['SalePrice']
x_train = feature_train[selected_columns]
print('Shape of x and y train : ', x_train.shape, y_train.shape)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
ordinal_encoder = OrdinalEncoder()
cat_ordinal_encoder = OrdinalEncoder()
cat_nominal_encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler(with_mean=False)
robust_scaler = RobustScaler(with_centering=False)
power_transform = PowerTransformer()
transformer = ColumnTransformer([('cat_nominal_encoder', cat_nominal_encoder, Categorical_nominal_columns)], remainder='passthrough')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
random_forest_regressor = RandomForestRegressor(random_state=0)
gradient_boost_regressor = GradientBoostingRegressor(random_state=0)
xgboost_regressor = XGBRegressor(random_state=0)
pipeline = Pipeline([('column_transformer', transformer), ('robust_scaler', robust_scaler), ('classifier', random_forest_regressor)])
params_classifier = [{'classifier': [random_forest_regressor], 'classifier__n_estimators': [100, 125], 'classifier__max_depth': [1, 3, 5, 7]}, {'classifier': [gradient_boost_regressor], 'classifier__loss': ['huber', 'ls', 'lad', 'quantile'], 'classifier__learning_rate': [0.08, 0.07, 0.05], 'classifier__n_estimators': [125], 'classifier__subsample': [0.6, 0.7, 0.8], 'classifier__max_depth': [3, 5, 7]}, {'classifier': [xgboost_regressor]}]
griseachcv = GridSearchCV(estimator=pipeline, param_grid=params_classifier, verbose=1, cv=10, scoring='neg_mean_squared_log_error', n_jobs=-1)