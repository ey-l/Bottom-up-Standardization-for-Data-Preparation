import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
train_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_path = '_data/input/house-prices-advanced-regression-techniques/test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train.head()
test.head()
print(f'Training Data shape: {train.shape} \n Test Data shape: {test.shape}')
cat_columns = train.select_dtypes(include=['object']).columns.tolist()
num_columns = train.select_dtypes(exclude=['object']).columns.tolist()
print(f'Categorical columns: \n {cat_columns} \n Numerical columns: \n {num_columns}')
num_columns = num_columns[:-1]
num_columns

def check_missing(df):
    missing = df.isna().sum()[df.isna().any() == True]
    df_out = pd.DataFrame({'missing': missing})
    return df_out
check_missing(train[cat_columns])
check_missing(train[num_columns])
X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=8)
X_train.head()
X_test.head()
test.head()
num_imputing = make_pipeline(SimpleImputer(strategy='constant', fill_value=0))
cat_imputing = make_pipeline(SimpleImputer(strategy='constant', fill_value='NA'))
ordinal_features = ['ExterQual', 'ExterCond', 'KitchenQual', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'CentralAir', 'LandSlope', 'PavedDrive', 'Fence', 'PoolQC', 'Alley', 'Street', 'Utilities']
nominal_features = list(set(cat_columns) - set(ordinal_features))
ql5 = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
fin = ['None', 'Unf', 'RFn', 'Fin']
expo = ['None', 'No', 'Mn', 'Av', 'Gd']
fint = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
func = ['None', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
yn = ['Y', 'N']
ls = ['None', 'Sev', 'Mod', 'Gtl']
pad = ['N', 'P', 'Y']
fen = ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
ql4 = ['None', 'Fa', 'TA', 'Gd', 'Ex']
al = ['None', 'Grvl', 'Pave']
st = ['None', 'Grvl', 'Pave']
util = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
ordinal_categories = [ql5, ql5, ql5, ql5, ql5, ql5, ql5, ql5, ql5, fin, expo, fint, fint, func, yn, ls, pad, fen, ql4, al, st, util]
len(ordinal_features) + len(nominal_features) + len(num_columns)
ordinal_enc = Pipeline(steps=[('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories))])
one_hot_enc = Pipeline(steps=[('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
imputing = ColumnTransformer(transformers=[('imp_nums', num_imputing, num_columns), ('imp_cats', cat_imputing, cat_columns)])
encoding = ColumnTransformer(transformers=[('enc_nums', 'passthrough', num_columns), ('enc_ord', ordinal_enc, ordinal_features), ('enc_nom', one_hot_enc, nominal_features)])
scaling = Pipeline(steps=[('scale', MinMaxScaler())])
cat_encoding = ColumnTransformer(transformers=[('enc_ord', ordinal_enc, ordinal_features), ('enc_nom', one_hot_enc, nominal_features)])
cats = Pipeline(steps=[('impute_cats', cat_imputing), ('encode_cats', one_hot_enc)])
nums = Pipeline(steps=[('impute_nums', num_imputing)])
preprocess = ColumnTransformer(transformers=[('cats', cats, cat_columns), ('nums', nums, num_columns)])
full_preprocess2 = Pipeline(steps=[('preprocess', preprocess), ('scaling', scaling)])
pd.DataFrame(full_preprocess2.fit_transform(X_train))
lm_pipeline = Pipeline(steps=[('full_preprocess', full_preprocess2), ('model', LinearRegression())])