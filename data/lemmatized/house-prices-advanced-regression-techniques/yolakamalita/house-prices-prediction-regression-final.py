import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
num_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd', 'SalePrice', 'MoSold', 'YrSold']
obj_cols = [col for col in _input1.columns if _input1[col].dtype == 'object' or col not in num_cols]
obj_cols
ordinal_cols = ['LotShape', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
nominal_cols = [col for col in obj_cols if col not in ordinal_cols]
nominal_cols
_input1['Alley'] = _input1['Alley'].astype(object).replace(np.nan, 'None')
_input1['MasVnrType'] = _input1['MasVnrType'].astype(object).replace(np.nan, 'None')
_input1['BsmtQual'] = _input1['BsmtQual'].astype(object).replace(np.nan, 'None')
_input1['BsmtCond'] = _input1['BsmtCond'].astype(object).replace(np.nan, 'None')
_input1['BsmtExposure'] = _input1['BsmtExposure'].astype(object).replace(np.nan, 'None')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].astype(object).replace(np.nan, 'None')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].astype(object).replace(np.nan, 'None')
_input1['FireplaceQu'] = _input1['FireplaceQu'].astype(object).replace(np.nan, 'None')
_input1['GarageFinish'] = _input1['GarageFinish'].astype(object).replace(np.nan, 'None')
_input1['GarageType'] = _input1['GarageType'].astype(object).replace(np.nan, 'None')
_input1['GarageQual'] = _input1['GarageQual'].astype(object).replace(np.nan, 'None')
_input1['GarageCond'] = _input1['GarageCond'].astype(object).replace(np.nan, 'None')
_input1['PoolQC'] = _input1['PoolQC'].astype(object).replace(np.nan, 'None')
_input1['Fence'] = _input1['Fence'].astype(object).replace(np.nan, 'None')
_input1['MiscFeature'] = _input1['MiscFeature'].astype(object).replace(np.nan, 'None')
_input0['Alley'] = _input0['Alley'].astype(object).replace(np.nan, 'None')
_input0['MasVnrType'] = _input0['MasVnrType'].astype(object).replace(np.nan, 'None')
_input0['BsmtQual'] = _input0['BsmtQual'].astype(object).replace(np.nan, 'None')
_input0['BsmtCond'] = _input0['BsmtCond'].astype(object).replace(np.nan, 'None')
_input0['BsmtExposure'] = _input0['BsmtExposure'].astype(object).replace(np.nan, 'None')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].astype(object).replace(np.nan, 'None')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].astype(object).replace(np.nan, 'None')
_input0['FireplaceQu'] = _input0['FireplaceQu'].astype(object).replace(np.nan, 'None')
_input0['GarageFinish'] = _input0['GarageFinish'].astype(object).replace(np.nan, 'None')
_input0['GarageType'] = _input0['GarageType'].astype(object).replace(np.nan, 'None')
_input0['GarageQual'] = _input0['GarageQual'].astype(object).replace(np.nan, 'None')
_input0['GarageCond'] = _input0['GarageCond'].astype(object).replace(np.nan, 'None')
_input0['PoolQC'] = _input0['PoolQC'].astype(object).replace(np.nan, 'None')
_input0['Fence'] = _input0['Fence'].astype(object).replace(np.nan, 'None')
_input0['MiscFeature'] = _input0['MiscFeature'].astype(object).replace(np.nan, 'None')
_input1['Electrical'] = _input1['Electrical'].astype(object).replace('-1', 'None')
_input0['Electrical'] = _input0['Electrical'].astype(object).replace('-1', 'None')
_input1['YearBuilt'] = 2010 - _input1['YearBuilt']
_input1['YearRemodAdd'] = 2010 - _input1['YearRemodAdd']
_input1['GarageYrBlt'] = 2010 - _input1['GarageYrBlt']
_input1['YrSold'] = 2010 - _input1['YrSold']
_input0['YearBuilt'] = 2010 - _input0['YearBuilt']
_input0['YearRemodAdd'] = 2010 - _input0['YearRemodAdd']
_input0['GarageYrBlt'] = 2010 - _input0['GarageYrBlt']
_input0['YrSold'] = 2010 - _input0['YrSold']
LotShape_cat = ['Reg', 'IR1', 'IR2', 'IR3']
LandSlope_cat = ['Gtl', 'Mod', 'Sev']
OverallQual_cat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
OverallCond_cat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ExterQual_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
ExterCond_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
BsmtQual_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
BsmtCond_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
BsmtExposure_cat = ['Gd', 'Av', 'Mn', 'No', 'None']
BsmtFinType1_cat = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None']
BsmtFinType2_cat = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None']
HeatingQC_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
KitchenQual_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
FireplaceQu_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
GarageQual_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
GarageCond_cat = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
PoolQC_cat = ['Ex', 'Gd', 'TA', 'Fa', 'None']
from scipy import stats

def finding_outlier(data):
    Median = np.median(data)
    data_L = data[data <= Median]
    data_U = data[data >= Median]
    C = 1.4826
    MAD_L = C * np.median(abs(data_L - Median))
    MAD_U = C * np.median(abs(data_U - Median))
    k = 4
    Lower = Median - k * MAD_L
    Upper = Median + k * MAD_U
    list_index = []
    list_outlier = []
    for i in range(len(data)):
        if data[i] < Lower or data[i] > Upper:
            list_index.append(i)
            list_outlier.append(data[i])
    return (list_index, list_outlier, Lower, Upper)
TotalBsmtSF = _input1['TotalBsmtSF']
(idx_TotalBsmtSF, val_TotalBsmtSF, Lower, Upper) = finding_outlier(TotalBsmtSF)
_1stFlrSF = _input1['1stFlrSF']
(idx_1stFlrSF, val_1stFlrSF, Lower, Upper) = finding_outlier(_1stFlrSF)
GrLivArea = _input1['GrLivArea']
(idx_GrLivArea, val_GrLivArea, Lower, Upper) = finding_outlier(GrLivArea)
GarageArea = _input1['GarageArea']
(idx_GarageArea, val_GarageArea, Lower, Upper) = finding_outlier(GarageArea)
YearBuilt = _input1['YearBuilt']
(idx_YearBuilt, val_YearBuilt, Lower, Upper) = finding_outlier(YearBuilt)
YearRemodAdd = _input1['YearRemodAdd']
(idx_YearRemodAdd, val_YearRemodAdd, Lower, Upper) = finding_outlier(YearRemodAdd)
outliers_mad = np.unique(idx_TotalBsmtSF + idx_1stFlrSF + idx_GrLivArea + idx_GarageArea + idx_YearBuilt + idx_YearRemodAdd)
from scipy.stats import multivariate_normal

def estimate_gaussian(dataset):
    mu = np.mean(dataset)
    sigma = np.cov(dataset.T)
    return (mu, sigma)

def multivariate_gaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def outlier_multivariate_gaussian(dataset, eps):
    (mu, sigma) = estimate_gaussian(dataset)
    p_mg = multivariate_gaussian(dataset, mu, sigma)
    eps = np.percentile(p_mg, eps)
    outliers = np.asarray(np.where(p_mg < eps))
    dataset_o = dataset.reset_index()
    dataset_o['outlier'] = dataset_o['index'].apply(lambda x: True if x in outliers else False)
    return (dataset_o, p_mg, eps, outliers)
TotalBsmtSF_ = _input1[['TotalBsmtSF', 'SalePrice']]
(TotalBsmtSF_, p_mg, eps, idx_TotalBsmtSF_) = outlier_multivariate_gaussian(TotalBsmtSF_, eps=0.7)
_1stFlrSF_ = _input1[['1stFlrSF', 'SalePrice']]
(_1stFlrSF_, p_mg, eps, idx_1stFlrSF_) = outlier_multivariate_gaussian(_1stFlrSF_, eps=0.8)
GrLivArea_ = _input1[['GrLivArea', 'SalePrice']]
(GrLivArea_, p_mg, eps, idx_GrLivArea_) = outlier_multivariate_gaussian(GrLivArea_, eps=0.8)
GarageArea_ = _input1[['GarageArea', 'SalePrice']]
(GarageArea_, p_mg, eps, idx_GarageArea_) = outlier_multivariate_gaussian(GarageArea_, eps=0.85)
YearBuilt_ = _input1[['YearBuilt', 'SalePrice']]
(YearBuilt_, p_mg, eps, idx_YearBuilt_) = outlier_multivariate_gaussian(YearBuilt_, eps=0.6)
YearRemodAdd_ = _input1[['YearRemodAdd', 'SalePrice']]
(YearRemodAdd_, p_mg, eps, idx_YearRemodAdd_) = outlier_multivariate_gaussian(YearRemodAdd_, eps=0.5)
outliers_mg = np.concatenate([idx_TotalBsmtSF_, idx_1stFlrSF_, idx_GrLivArea_, idx_GarageArea_, idx_YearBuilt_, idx_YearRemodAdd_], axis=1)
outliers_mg = np.unique(outliers_mg)
outliers_removal = np.concatenate([outliers_mad, outliers_mg[~np.isin(outliers_mg, outliers_mad)]])
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
train_no = _input1.drop(outliers_removal)
y = train_no['SalePrice']
X = train_no.drop(['SalePrice'], axis=1)
y_log = np.log(y)
numerical_col = [col for col in X.columns if col in num_cols]
nominal_col = [col for col in X.columns if col in nominal_cols]
ordinal_col = [col for col in X.columns if col in ordinal_cols]
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, data=X, target=y_log):
    (X_train, X_valid, y_train, y_valid) = train_test_split(data, target, train_size=0.9, test_size=0.1, random_state=42)
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaling', RobustScaler())])
    nominal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ordinalencoder', OrdinalEncoder(categories=[LotShape_cat, LandSlope_cat, OverallQual_cat, OverallCond_cat, ExterQual_cat, ExterCond_cat, BsmtQual_cat, BsmtCond_cat, BsmtExposure_cat, BsmtFinType1_cat, BsmtFinType2_cat, HeatingQC_cat, KitchenQual_cat, FireplaceQu_cat, GarageQual_cat, GarageCond_cat, PoolQC_cat]))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_col), ('nom', nominal_transformer, nominal_col), ('ord', ordinal_transformer, ordinal_col)])
    param = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1), 'max_depth': trial.suggest_int('max_depth', 10, 100), 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 70), 'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.1, 0.25, 0.5, 0.75, 1.0]), 'feature_fraction': trial.suggest_categorical('feature_fraction', [0.1, 0.25, 0.5, 0.75, 1.0]), 'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 0.1, 1.0), 'device_type': 'gpu'}
    model = lgb.LGBMRegressor(**param)
    pipeline_lgbm = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])