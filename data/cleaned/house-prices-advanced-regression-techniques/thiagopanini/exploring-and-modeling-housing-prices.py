

import pandas as pd
import os
from warnings import filterwarnings
filterwarnings('ignore')
DATA_PATH = '_data/input/house-prices-advanced-regression-techniques/'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()
from xplotter.insights import *
TARGET = 'SalePrice'
df_overview = data_overview(df=df, corr=True, target=TARGET)
print(f'Some of the features and its metadata')
df_overview.head(25)
df_overview.sort_values(by='qtd_cat', ascending=False).head()
df_overview.sort_values(by='target_pearson_corr', ascending=False).head()
space_cols = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageType', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MoSold', 'OpenPorchSF', 'PoolArea', 'SaleCondition', 'SaleType', 'ScreenPorch', 'TotalBsmtSF', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold']
num_space_cols = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
cat_space_cols = ['BsmtQual', 'GarageType', 'SaleType', 'SaleCondition', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']
plot_multiple_distplots(df=df, col_list=num_space_cols, kind='boxen')
normal_dist_num_space_cols = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
plot_multiple_dist_scatterplot(df=df, col_list=normal_dist_num_space_cols, y_col='SalePrice')
plot_multiple_countplots(df=df, col_list=cat_space_cols, orient='v')
plot_cat_aggreg_report(df=df, cat_col='BsmtQual', value_col=TARGET, title3='Statistical Analysis', desc_text=f'A statistical approach for {TARGET} \nusing the data available', stat_title_mean='Mean', stat_title_median='Median', stat_title_std='Std')
plot_cat_aggreg_report(df=df, cat_col='GarageCars', value_col=TARGET, title3='Statistical Analysis', desc_text=f'A statistical approach for {TARGET} \nusing the data available', stat_title_mean='Mean', stat_title_median='Median', stat_title_std='Std', dist_kind='box')
plot_cat_aggreg_report(df=df, cat_col='SaleCondition', value_col=TARGET, title3='Statistical Analysis', desc_text=f'A statistical approach for {TARGET} \nusing the data available', stat_title_mean='Mean', stat_title_median='Median', stat_title_std='Std', dist_kind='strip')
plot_cat_aggreg_report(df=df, cat_col='Fireplaces', value_col=TARGET, title3='Statistical Analysis', desc_text=f'A statistical approach for {TARGET} \nusing the data available', stat_title_mean='Mean', stat_title_median='Median', stat_title_std='Std', dist_kind='boxen')
plot_evolutionplot(df=df, x='YrSold', y='SalePrice', agg_type='sum', date_col=False, x_rot=0, label_aggreg='M')
plot_evolutionplot(df=df, x='MoSold', y='SalePrice', agg_type='sum', date_col=False, x_rot=0, label_aggreg='M')
df_tmp = df.copy()
df_tmp['YrMoSold'] = df_tmp['YrSold'] * 100 + df_tmp['MoSold']
plot_evolutionplot(df=df_tmp, x='YrMoSold', y='SalePrice', agg_type='sum', date_col=False, label_aggreg='M')
plot_evolutionplot(df=df_tmp, x='YrMoSold', y='SalePrice', hue='SaleCondition', agg_type='mean', label_data=False, style='SaleCondition', palette='plasma')
plot_evolutionplot(df=df_tmp, x='YrMoSold', y='SalePrice', hue='Fireplaces', agg_type='mean', date_col=False, label_data=False, style='Fireplaces')
building_cols = ['MSSubClass', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoodMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']
num_building_cols = ['MasVnrArea', 'MiscVal', 'BsmtFinSF2', 'MSSubClass', 'BsmtUnfSF', 'BsmtFinSF1']
cat_building_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtFinType1', 'MasVnrType', 'Electrical', 'Functional', 'KitchenQual', 'PavedDrive', 'HeatingQC', 'LandSlope', 'HouseStyle', 'BldgType', 'LotConfig', 'Utilities', 'LandContour', 'LotShape', 'Street', 'CentralAir', 'Heating', 'RoofStyle', 'Foundation', 'ExterCond', 'ExterQual', 'Exterior2nd', 'Exterior1st', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'GarageYrBlt', 'YearBuilt']
plot_multiple_distplots(df=df, col_list=num_building_cols, kind='strip')
plot_multiple_countplots(df=df, col_list=cat_building_cols)
plot_multiple_dist_scatterplot(df=df, col_list=['BsmtUnfSF', 'BsmtFinSF1'], y_col='SalePrice')
plot_cat_aggreg_report(df=df, cat_col='ExterQual', value_col=TARGET, dist_kind='boxen')
plot_cat_aggreg_report(df=df, cat_col='KitchenQual', value_col=TARGET, dist_kind='strip')
plot_cat_aggreg_report(df=df, cat_col='GarageQual', value_col=TARGET, dist_kind='box')
location_cols = ['MSZoning', 'Neighborhood', 'Condition1', 'Condition2']
plot_cat_aggreg_report(df=df, cat_col='MSZoning', value_col=TARGET, dist_kind='boxen')
TO_DROP = ['Condition2', 'RoofMatl']
(fig, axs) = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))
plot_countplot(df=df, col='Condition2', ax=axs[0])
plot_countplot(df=df, col='RoofMatl', ax=axs[1])
from mlcomposer.transformers import ColumnSelection
df_tmp = df.copy()
TARGET = 'SalePrice'
TO_DROP = ['Condition2', 'RoofMatl', 'Id']
INITIAL_FEATURES = [col for col in df_tmp.columns if col not in TO_DROP]
selector = ColumnSelection(features=INITIAL_FEATURES)
df_slct = selector.fit_transform(df_tmp)
print(f'Shape before feature selection: {df_tmp.shape}')
print(f'Shape after feature selection: {df_slct.shape}')
from mlcomposer.transformers import CategoricalLimitter
from sklearn.pipeline import Pipeline
N_CAT = 5
HIGH_CAT_FEATURES = data_overview(df=df_tmp).query('qtd_cat > @N_CAT + 1')['feature'].values
HIGH_CAT_FEATURES = [col for col in HIGH_CAT_FEATURES if col not in TO_DROP]
CAT3_FEATURES = ['Functional', 'SaleType']
CAT8_FEATURES = ['Neighborhood']
CAT5_FEATURES = ['HouseStyle', 'Condition1', 'Exterior2nd', 'Exterior1st']
OTHER_TAG = 'Other'
cat3_agrup = CategoricalLimitter(features=CAT3_FEATURES, n_cat=3, other_tag=OTHER_TAG)
cat5_agrup = CategoricalLimitter(features=CAT5_FEATURES, n_cat=5, other_tag=OTHER_TAG)
cat8_agrup = CategoricalLimitter(features=CAT8_FEATURES, n_cat=8, other_tag=OTHER_TAG)
cat_agrup_pipeline = Pipeline([('cat3_agrup', cat3_agrup), ('cat5_agrup', cat5_agrup), ('cat8_agrup', cat8_agrup)])
df_cat_agrup = cat_agrup_pipeline.fit_transform(df_slct)
plot_multiple_countplots(df=df_tmp, col_list=HIGH_CAT_FEATURES)
plot_multiple_countplots(df=df_cat_agrup, col_list=HIGH_CAT_FEATURES)
from mlcomposer.transformers import DropDuplicates
dup_dropper = DropDuplicates()
df_nodup = dup_dropper.fit_transform(df_cat_agrup)
print(f'Total of duplicates before: {df_cat_agrup.duplicated().sum()}')
print(f'Total of duplicates after: {df_nodup.duplicated().sum()}')
plot_distplot(df=df, col=TARGET, hist=True, title=f'{TARGET} distribution plot')
from mlcomposer.transformers import LogTransformation
(fig, axs) = plt.subplots(nrows=1, ncols=2, figsize=(17, 7))
plot_distplot(df=df, col=TARGET, hist=True, ax=axs[0], title=f'Original {TARGET} distribution')
log_tr = LogTransformation(cols_to_log=TARGET)
df_target_log = log_tr.fit_transform(df)
plot_distplot(df=df_target_log, col=TARGET, hist=True, color='mediumseagreen', ax=axs[1], title=f'{TARGET} distribution after log transformation')
from mlcomposer.transformers import DataSplitter
splitter = DataSplitter(target=TARGET)
(X_train, X_test, y_train, y_test) = splitter.fit_transform(df_nodup)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')
from sklearn.pipeline import Pipeline
tmp_pipeline = Pipeline([('selector', ColumnSelection(features=INITIAL_FEATURES)), ('cat_agrup', Pipeline([('cat3_agrup', CategoricalLimitter(features=CAT3_FEATURES, n_cat=3, other_tag=OTHER_TAG)), ('cat5_agrup', CategoricalLimitter(features=CAT5_FEATURES, n_cat=5, other_tag=OTHER_TAG)), ('cat8_agrup', CategoricalLimitter(features=CAT8_FEATURES, n_cat=8, other_tag=OTHER_TAG))])), ('splitter', DataSplitter(target=TARGET))])
(X_train, X_test, y_train, y_test) = tmp_pipeline.fit_transform(df)
print(f'Shape of original dataset: {df.shape}')
print(f'Shape of X_train: {X_train.shape}')
high_cat_dict = {}
for feature in HIGH_CAT_FEATURES:
    high_cat_dict[feature] = [col for col in X_train[feature].value_counts().index if col != OTHER_TAG]
high_cat_dict
from mlcomposer.transformers import CategoricalMapper
initial_pipeline = Pipeline([('selector', ColumnSelection(features=INITIAL_FEATURES)), ('cat_agrup', CategoricalMapper(cat_dict=high_cat_dict, other_tag=OTHER_TAG)), ('splitter', DataSplitter(target=TARGET))])
(X_train, X_test, y_train, y_test) = initial_pipeline.fit_transform(df)
print(f'Shape of original dataset: {df.shape}')
print(f'Shape of X_train: {X_train.shape}')
plot_multiple_countplots(df=X_train, col_list=HIGH_CAT_FEATURES)
num_features = [col for (col, dtype) in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for (col, dtype) in X_train.dtypes.items() if dtype == 'object']
X_train_num = X_train[num_features]
X_cat_num = X_train[cat_features]
print(f'Total of numerical features: {len(num_features)}')
print(f'Total of categorical features: {len(cat_features)}')
print(f'Total of features (must be sum of numerical and categorical ones): {X_train.shape[1]}')
df_overview.query('feature in @num_features').head()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_num_filled = imputer.fit_transform(X_train_num)
print(f'Null data before pipeline: {X_train_num.isnull().sum().sum()}')
print(f'Null data after pipeline: {pd.DataFrame(X_train_num_filled, columns=num_features).isnull().sum().sum()}')
col_log = 'LotArea'
tmp = df.copy()
tmp[col_log] = np.log1p(tmp[col_log])
(fig, axs) = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))
plot_distplot(df=df, col=col_log, hist=True, ax=axs[0], title=f'Distplot {col_log} - Original')
plot_distplot(df=tmp, col=col_log, hist=True, ax=axs[1], title=f'Distplot {col_log} - Log Transformation')
from scipy.stats import skew, kurtosis
tmp_ov = df_overview.copy()
tmp_ov['skew'] = tmp_ov.query('feature in @num_features')['feature'].apply(lambda x: skew(X_train_num[x]))
tmp_ov['kurtosis'] = tmp_ov.query('feature in @num_features')['feature'].apply(lambda x: kurtosis(X_train_num[x]))
tmp_ov[~tmp_ov['skew'].isnull()].sort_values(by='skew', ascending=False).loc[:, ['feature', 'skew', 'kurtosis']]
from mlcomposer.transformers import DynamicLogTransformation
SKEW_THRESH = 0.5
cols_idx = [np.argwhere(skew(X_train_num) == sk)[0][0] for sk in skew(X_train_num) if sk > SKEW_THRESH]
cols_to_log = list(X_train_num.iloc[:, cols_idx].columns)
print(f'First line before transformation: \n\n{X_train_num_filled[0]}')
log_tr = DynamicLogTransformation(num_features=num_features, cols_to_log=cols_to_log)
X_train_num_log = log_tr.fit_transform(X_train_num_filled)
print(f'\nFirst line after transformation: \n\n{X_train_num_log[0]}')
from sklearn.preprocessing import StandardScaler
from mlcomposer.transformers import DynamicScaler
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_features)
X_train_num_scaled.head()
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('log', DynamicLogTransformation(num_features=num_features, cols_to_log=cols_to_log)), ('scaler', DynamicScaler(scaler_type='Standard'))])
X_train_num_prep = num_pipeline.fit_transform(X_train_num)
print(f'Shape before num_pipeline: {X_train_num.shape}')
print(f'Shape after num_pipeline:{X_train_num_prep.shape}')
print(f'\nX_train_num[0]:\n{np.array(X_train_num.iloc[0, :])}')
print(f'\nX_train_num_prep[0]:\n{X_train_num_prep[0]}')
print(f'Total of categorical features: {len(cat_features)}')
print(f'Example of categorical training data:')
X_train_cat = X_train[cat_features]
X_train_cat.head()
cat_overview = data_overview(df=X_train_cat)
cat_overview.head(10)
from mlcomposer.transformers import DummiesEncoding
encoder = DummiesEncoding(dummy_na=True)
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
print(f'Shape before encoding: {X_train_cat.shape}')
print(f'Shape after encoding: {X_train_cat_encoded.shape}')
X_train_cat_encoded.head()
cat_pipeline = Pipeline([('encoder', DummiesEncoding(dummy_na=True))])
X_train_cat_encoded = cat_pipeline.fit_transform(X_train_cat)
print(f'Shape before cat_pipeline: {X_train_cat.shape}')
print(f'Shape after cat_pipeline: {X_train_cat_encoded.shape}')
TARGET = 'SalePrice'
TO_DROP = ['Condition2', 'RoofMatl', 'Id']
INITIAL_FEATURES = [col for col in df_tmp.columns if col not in TO_DROP]
CAT_GROUP_DICT = {'Functional': ['Typ', 'Min2', 'Min1'], 'SaleType': ['WD', 'New', 'COD'], 'HouseStyle': ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'], 'Condition1': ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN'], 'Neighborhood': ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer'], 'Exterior2nd': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'Plywood'], 'Exterior1st': ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']}
OTHER_TAG = 'Other'
COLS_TO_LOG = ['MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
NUM_FEATURES = [col for (col, dtype) in X_train.dtypes.items() if dtype != 'object']
CAT_FEATURES = [col for (col, dtype) in X_train.dtypes.items() if dtype == 'object']
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
initial_prep_pipeline = Pipeline([('col_filter', ColumnSelection(features=INITIAL_FEATURES)), ('cat_agrup', CategoricalMapper(cat_dict=CAT_GROUP_DICT, other_tag=OTHER_TAG)), ('log_target', LogTransformation(cols_to_log=TARGET))])
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('log', DynamicLogTransformation(num_features=NUM_FEATURES, cols_to_log=COLS_TO_LOG)), ('scaler', DynamicScaler(scaler_type=None))])
cat_pipeline = Pipeline([('encoder', DummiesEncoding(dummy_na=True))])
prep_pipeline = ColumnTransformer([('num', num_pipeline, NUM_FEATURES), ('cat', cat_pipeline, CAT_FEATURES)])
df_train = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df_train_prep = initial_prep_pipeline.fit_transform(df_train)
(X_train, X_val, y_train, y_val) = train_test_split(df_train_prep.drop(TARGET, axis=1), df_train_prep[TARGET].values, test_size=0.2, random_state=42)
X_train_prep = prep_pipeline.fit_transform(X_train)
train_features = prep_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding
X_val_prep = prep_pipeline.fit_transform(X_val)
val_features = prep_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding
print(f'Shape of X_train_prep: {X_train_prep.shape}')
print(f'Shape of X_val_prep: {X_val_prep.shape}')
print(f'Total of train_features (after encoding): {len(train_features)}')
print(f'Total of val_features (after encoding): {len(val_features)}')
print(f'Difference between train and test: {len(train_features) - len(val_features)}')
not_included_val = [col for col in train_features if col not in val_features]
print(f'\nCategorical entries included on X_train but not in X_val:')
not_included_val
X_train_prep_df = pd.DataFrame(X_train_prep, columns=num_features + train_features)
MODEL_FEATURES = list(X_train_prep_df.columns)
X_val_prep_df = pd.DataFrame(X_val_prep, columns=num_features + val_features)
for col in not_included_val:
    X_val_prep_df[col] = 0
X_val_prep = np.array(X_val_prep_df.loc[:, MODEL_FEATURES])
print(f'Shape of new X_train_prep: {X_train_prep.shape}')
print(f'Shape of new X_val_prep: {X_val_prep.shape}')
print(f'Total of model features: {len(MODEL_FEATURES)}')
X_val_prep_df = X_val_prep_df.loc[:, MODEL_FEATURES]
print(f'\nLast 5 features of X_train_prep: \n{X_train_prep_df.iloc[:, -5:].columns}')
print(f'\nLast 5 features of X_val_prep: \n{X_val_prep_df.iloc[:, -5:].columns}')
X_train_prep = np.array(X_train_prep_df)
X_val_prep = np.array(X_val_prep_df)
print(f'\nShape of final X_train_prep: {X_train_prep.shape}')
print(f'Shape of final X_val_prep: {X_val_prep.shape}')
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
ridge_reg = Ridge()
lasso_reg = Lasso()
elastic_reg = ElasticNet()
lgbm_reg = LGBMRegressor(objective='regression')
xgb_reg = XGBRegressor(objective='reg:squarederror')
lin_reg_params = {'fit_intercept': [True, False], 'normalize': [True, False]}
tree_reg_params = {'max_depth': [100, 200, 300, 350, 400, 500], 'max_features': np.arange(1, len(MODEL_FEATURES)), 'random_state': [42]}
forest_reg_params = {'n_estimators': [75, 90, 100, 200, 300, 400, 450, 500], 'max_features': np.arange(1, len(MODEL_FEATURES)), 'random_state': [42]}
ridge_reg_params = {'alpha': np.linspace(1e-05, 20, 400), 'fit_intercept': [True, False], 'normalize': [True, False]}
lasso_reg_params = {'alpha': np.linspace(1e-05, 20, 400), 'fit_intercept': [True, False], 'normalize': [True, False]}
elastic_reg_params = {'alpha': np.linspace(1e-05, 20, 400), 'l1_ratio': np.linspace(0, 1, 400), 'fit_intercept': [True, False], 'normalize': [True, False]}
lgbm_param_grid = {'num_leaves': np.arange(10, 250, 1), 'max_depth': np.arange(10, 350, 1), 'n_estimators': [75, 90, 100, 200, 300, 400, 450, 500], 'learning_rate': np.linspace(1e-05, 20, 400), 'reg_alpha': np.linspace(1e-05, 20, 400), 'reg_lambda': np.linspace(1e-05, 20, 400)}
xgb_param_grid = {'reg_lambda': np.linspace(1e-05, 20, 400), 'reg_alpha': np.linspace(1e-05, 20, 400), 'max_depth': np.arange(10, 350, 1), 'n_estimators': [75, 90, 100, 200, 300, 400, 450, 500], 'random_state': [42]}
set_regressors = {'LinearRegression': {'model': lin_reg, 'params': lin_reg_params}, 'DecisionTreeRegressor': {'model': tree_reg, 'params': tree_reg_params}, 'RandomForestRegressor': {'model': forest_reg, 'params': forest_reg_params}, 'Ridge': {'model': ridge_reg, 'params': ridge_reg_params}, 'Lasso': {'model': lasso_reg, 'params': lasso_reg_params}, 'ElasticNet': {'model': elastic_reg, 'params': elastic_reg_params}, 'LightGBM': {'model': lgbm_reg, 'params': {}}, 'XGBoost': {'model': xgb_reg, 'params': {}}}
from mlcomposer.trainer import LinearRegressor
trainer = LinearRegressor()