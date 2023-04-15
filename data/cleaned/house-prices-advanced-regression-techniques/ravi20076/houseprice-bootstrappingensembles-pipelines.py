import numpy as np
from scipy.stats import iqr
import pandas as pd
import re
from datetime import date
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
from termcolor import colored
from warnings import filterwarnings
filterwarnings(action='ignore')
from tqdm.notebook import tqdm
from sklearn_pandas import DataFrameMapper, gen_features, NumericalTransformer
from sklearn_pandas.pipeline import Pipeline as skpPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, mutual_info_regression
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
target = 'SalePrice'
null_cutoff = 0.1
skew_cutoff = 0.5
feat_sel_threshold = 0.0
train_frac = 0.95
xytrain = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', encoding='utf8')
xtest = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', encoding='utf8')
sub_fl = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv', encoding='utf8')
print(colored(f'Train data columns are \n{xytrain.columns}', color='blue'))
print(colored(f'\nTrain data details\n', color='blue', attrs=['dark', 'bold']))
xytrain.info()
print(colored(f'\nData preprocessing and pre-pipeline steps\n', color='blue', attrs=['dark', 'bold']))
(xtrain, ytrain) = (xytrain.drop([target, 'Id'], axis=1), xytrain[[target]])
xtest.drop('Id', axis=1, errors='ignore', inplace=True)
num_enc_feat_lst = ['OverallQual', 'OverallCond', 'MSSubClass']
xtrain[num_enc_feat_lst] = xtrain[num_enc_feat_lst].astype(str)
xtest[num_enc_feat_lst] = xtest[num_enc_feat_lst].astype(str)
null_feat_lst = xtrain.isna().sum(axis=0)
null_feat_lst = null_feat_lst[null_feat_lst > 0].sort_values(ascending=False)
num_null_feat_drop_lst = null_feat_lst[null_feat_lst > null_cutoff * xtrain.index.max()].index
xtrain.drop(num_null_feat_drop_lst, axis=1, inplace=True, errors='ignore')
xtest.drop(num_null_feat_drop_lst, axis=1, inplace=True, errors='ignore')
print(colored('\nNull features in the training set\n', color='blue', attrs=['bold', 'dark']))

print(colored(f'\nFeatures with nulls > {null_cutoff:.2%} to be dropped are \n{list(num_null_feat_drop_lst)}\n', color='blue', attrs=['bold', 'dark']))
print('\n')
(fig, ax) = plt.subplots(1, 1, figsize=(7, 7))
ax = null_feat_lst.plot.bar(color='tab:blue')
ax.set_title('Null features in the training set', color='tab:blue', fontsize=16)
ax.axhline(y=round(null_cutoff * xtrain.index.max()), color='darkred', linewidth=1.5)
ax.set_ylabel('Null values', color='tab:blue', fontsize=8)
ax.set_xlabel('Features', color='tab:blue', fontsize=8)
plt.xticks(color='tab:blue', fontsize=8, rotation=90)
plt.yticks(fontsize=8, color='tab:blue')
plt.tight_layout()

del fig, ax
num_feat_lst = list(xtrain.head(1).select_dtypes(include=np.number).columns)
char_feat_lst = list(xtrain.head(1).select_dtypes(exclude=np.number).columns)
print(colored(f'\nTraining features after dropping null columns are-\n', color='blue', attrs=['dark', 'bold']))
print(colored(f'\nNumeric training features are-\n{num_feat_lst}\n', color='blue', attrs=['dark']))
print(colored(f'\nObject training features are-\n{char_feat_lst}\n', color='blue', attrs=['dark']))
print(colored(f'\nTraining set description\n', color='blue', attrs=['bold', 'dark']))

print('\n')
(fig, ax) = plt.subplots(1, 2, figsize=(20, 8))
sns.distplot(ytrain[target].values, color='tab:blue', ax=ax[0])
sns.distplot(np.log(ytrain[target]), color='tab:blue', ax=ax[1])
ax[0].set_title('Target column distribution without logarithm transform', color='tab:blue', fontsize=16, loc='center')
ax[1].set_title('Target column distribution with logarithm transform', color='tab:blue', fontsize=16, loc='center')
plt.tight_layout()

del fig, ax
char_fill_none_feat = ['MasVnrType']
char_mode_fill_feat = [col for col in char_feat_lst if col != 'MasVnrType']
num_fill_0_feat = ['BsmtFinSF1', 'BsmtFinSF2', 'HalfBath', 'Fireplaces', 'OpenPorchSF', 'GrLivArea', 'BedroomAbvGr', 'EnclosedPorch', 'BsmtFullBath', 'FullBath', 'KitchenAbvGr', 'GarageCars', '3SsnPorch', 'MasVnrArea', '1stFlrSF', 'BsmtHalfBath', 'ScreenPorch', '2ndFlrSF', 'WoodDeckSF', 'PoolArea']
num_fill_median_feat = ['LotArea', 'LowQualFinSF', 'MiscVal', 'BsmtUnfSF', 'GarageArea', 'MoSold', 'YearBuilt']
num_passthrough_feat = ['TotRmsAbvGrd', 'GarageYrBlt', 'YearRemodAdd', 'TotalBsmtSF', 'YrSold']
new_feat_lst = ['Tot_Baths', 'Prop_Area', 'Porch_WdDeck', 'Pool_Fl', 'Prop_BldSell_Cyc', 'Is_Remodelled', 'Is_ReMdlB4Sale']
std_feat_lst = ['BsmtFinSF1', 'BsmtFinSF2', 'OpenPorchSF', 'GrLivArea', 'MasVnrArea', '2ndFlrSF', 'WoodDeckSF', 'PoolArea', 'LotArea', 'LowQualFinSF', 'MiscVal', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF', 'Prop_Area', 'Prop_BldSell_Cyc']
feat_lst = char_fill_none_feat + char_mode_fill_feat + num_fill_0_feat + num_fill_median_feat + num_passthrough_feat
dt_feat_lst = [col for col in num_feat_lst if re.findall('yr|year|mo|sold', col.lower()) != []]
print(colored(f'Total unique segmented columns are- {len(set(feat_lst))}\n', color='blue', attrs=['bold', 'dark']))
print(colored(f'Total features in the training data are- {len(num_feat_lst + char_feat_lst)}\n', color='red', attrs=['bold', 'dark']))
print(colored(f'Year-date features in the training data are-\n{dt_feat_lst}\n', color='blue', attrs=['bold', 'dark']))
xtrain = xtrain[feat_lst]
xtest = xtest[feat_lst]
print(colored(f'\nNewly ordered feature list is \n', color='blue', attrs=['dark', 'bold']))
print(colored(f'{feat_lst}\n', color='blue'))

def Make_Features(df, feat_lst=feat_lst):
    """
    This function creates new features from the existing features to improve predictablity
    1. Total bathrooms
    2. Total basement square feet area (null treated)
    3. Property total area
    4. Pool flag
    5. Property build to sale period- this is clipped to 0 and positive values to prevent non-intuitive negative values
    6. Remodelled flag
    7. Remodelled before sale flag
    8. Fireplace and wood-porch flag 
    9. Null filling for garage year built with year-built
    This function is used with the sklearn function transformer as it is a stateless transformation
    
    Inputs- df (dataframe):- null treated and encoded dataframe from the previous pipeline step
    Returns- df (dataframe):- new dataframe with the new features
    """
    df['Tot_Baths'] = df.FullBath + df.BsmtFullBath + (df.HalfBath + df.BsmtHalfBath) * 0.5
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF'])
    df['Prop_Area'] = df.TotalBsmtSF + df.WoodDeckSF + df.GrLivArea + df['3SsnPorch'] + df.OpenPorchSF + df.ScreenPorch + df.EnclosedPorch + df.MasVnrArea + df.GarageArea + df.PoolArea
    df['Porch_WdDeck'] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df.WoodDeckSF + df['3SsnPorch']
    df['Pool_Fl'] = np.where(df.PoolArea > 0, 1, 0)
    df['YrSold'] = df['YrSold'].fillna(date.today().year - 1)
    df['Prop_BldSell_Cyc'] = np.clip(df['YrSold'] - df['YearBuilt'], a_min=0.0, a_max=None)
    df['Is_Remodelled'] = np.where(df.YearRemodAdd != df.YearBuilt, 1, 0)
    df['Is_ReMdlB4Sale'] = np.where(abs(df['YrSold'] - df['YearRemodAdd']) <= 1, 1, 0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    _ = df.isna().sum(axis=0)
    print(colored(f'\nNull columns after processing and feature addition\n', color='blue', attrs=['bold', 'dark']))

    print('\n')
    del _
    return df

class DtColDropper(BaseEstimator, TransformerMixin):
    """This class drops the data columns from the dataset after all feature processing is done."""

    def __init__(self, cols=None):
        """This function initialises the date columns from the input- dt_feat_lst global variable for the drop"""
        if not isinstance(cols, list):
            self.cols = cols
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        return X1.loc[:, ~X1.columns.isin(self.cols)]

class SkewVarXformer(BaseEstimator, TransformerMixin):
    """
    This class is a part of the overall processing pipeline that removes skewness from numerical variables.
    It checks for skewed numeric variables and tries to reduce it by taking log of skewed variables.
    """

    def __init__(self, skew_cutoff=skew_cutoff):
        self.skew_cutoff = skew_cutoff

    def fit(self, X, y=None, **fit_params):
        """
        This function calculates the skewness for all variables in the dataset
        
        Inputs- 
        self- current state of the class
        X,y (dataframe):- Input dataframe for the function
        fit_params (dict):- keyword arguments for the function, if desired
        
        Returns- 
        self- current state of the class (learns the skewness of numeric columns)
        """
        global char_feat_lst
        self.skew_val = X.iloc[:, len(char_feat_lst):].skew()
        return self

    def transform(self, X, y=None, **transform_param):
        """
        This function transforms highly skewed variables with the log transform
        
        Inputs- 
        self- current state of the class
        X,y (dataframe):- Input dataframe for the function
        skew_cutoff (float):- cutoff for the skewness for logarithm transform
        transform_params (dict):- keyword arguments for the function, if desired 
        
        Returns- 
        X1 (dataframe):- Dataframe with the modified values         
        """
        X1 = X.copy()
        xform_col_lst = list(self.skew_val.loc[abs(self.skew_val) > self.skew_cutoff].index)
        X1[xform_col_lst] = np.log1p(X1[xform_col_lst].values)
        return X1

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class is designed to elicit high univariate-performing independent variables with the target and shortlist such variables.
    2 selection methods are used- corr (correlation) and mutual_info (mutual information regression)
    """

    def __init__(self, feat_sel_threshold: float, sel_mthd_lbl: str):
        self.feat_sel_threshold = feat_sel_threshold
        self.sel_mthd_lbl = sel_mthd_lbl.lower()

    def fit(self, X, y, target=target, **fit_params):
        """
        This function calculates the Pearson correlation/ mutual info regression between the features and the target column in the training data        
        """
        if self.sel_mthd_lbl == 'corr':
            xy = pd.concat((X, y), axis=1)
            self.feat_sel_mtrc = xy.corr().drop([target], axis=0)[[target]]
        elif self.sel_mthd_lbl == 'mutual_info':
            self.feat_sel_mtrc = pd.DataFrame(mutual_info_regression(X, y, random_state=10), index=X.columns, columns=[target])
        return self

    def transform(self, X, y=None, target=target, **transform_params):
        """
        This function selects the high-performing columns and retains them in the relevant data-set
        """
        X1 = X.copy()
        sel_feat_lst = list(self.feat_sel_mtrc.loc[abs(self.feat_sel_mtrc[target]) >= self.feat_sel_threshold, :].index)
        return X1.loc[:, sel_feat_lst]

class TargetXformer(BaseEstimator, TransformerMixin):
    """This class transforms the target column in the train set and then defines the inverse transform for the predictions"""

    def __init__(self, xform_func_lbl: str):
        self.xform_func_lbl = xform_func_lbl.lower()

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame, y=None, target=target, **transform_params):
        """This function transforms the target column"""
        X1 = X.copy()
        if self.xform_func_lbl == 'log':
            X1['xform'] = np.log(X1[target])
            X1.drop(target, axis=1, inplace=True)
            X1.rename({'xform': target}, axis=1, inplace=True)
        return X1

    def inverse_transform(self, X, y=None, target=target, **inv_xform_params):
        """This function inverts the transform based on the transform function used"""
        X1 = X.copy()
        if self.xform_func_lbl == 'log':
            X1 = np.exp(X1)
        return X1
fill_mode_ord_enc_feat = gen_features(columns=[col.split(' ') for col in char_mode_fill_feat], classes=[{'class': SimpleImputer, 'strategy': 'most_frequent'}, {'class': OrdinalEncoder, 'handle_unknown': 'use_encoded_value', 'unknown_value': 99}], suffix={})
fill_0_num = gen_features(columns=[col.split(' ') for col in num_fill_0_feat], classes=[{'class': SimpleImputer, 'strategy': 'constant'}], suffix={})
fill_median_num = gen_features(columns=[col.split(' ') for col in num_fill_median_feat], classes=[{'class': SimpleImputer, 'strategy': 'median'}], suffix={})
null_trmt_ord_enc = skpPipeline([('fill_None_mapper', DataFrameMapper(default=None, df_out=True, input_df=True, features=[(char_fill_none_feat, [SimpleImputer(strategy='constant', fill_value='None'), OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99)])])), ('fill_mode_mapper', DataFrameMapper(features=fill_mode_ord_enc_feat, default=None, df_out=True, input_df=True)), ('fill_0_mapper', DataFrameMapper(features=fill_0_num, default=None, df_out=True, input_df=True)), ('fill_median_mapper', DataFrameMapper(features=fill_median_num, default=None, df_out=True, input_df=True))])
std_num_feat = DataFrameMapper(gen_features(columns=[col.split(' ') for col in std_feat_lst], classes=[StandardScaler], suffix={}), default=None, df_out=True, input_df=True)
y_processor = Pipeline([('target_xform', TargetXformer(xform_func_lbl='log')), ('tgt_zscorer', StandardScaler())])
x_procesor = Pipeline([('null_trmt_ord_enc', null_trmt_ord_enc), ('new_feat_dev', FunctionTransformer(Make_Features)), ('drop_dt_feat', DtColDropper(cols=dt_feat_lst)), ('skew_var_xform', SkewVarXformer(skew_cutoff=skew_cutoff)), ('z_scorer', std_num_feat), ('feat_sel', FeatureSelector(feat_sel_threshold=feat_sel_threshold, sel_mthd_lbl='corr'))])
print(colored(f'Training set pipeline invocation:-', color='red', attrs=['bold', 'dark']))
y1 = pd.DataFrame(y_processor.fit_transform(ytrain), columns=[target])
x1 = x_procesor.fit_transform(xtrain, y1)
print(colored(f'Test set pipeline invocation:-', color='red', attrs=['bold', 'dark']))
xt = x_procesor.transform(xtest)
print(colored(f'Dataframes x1-y1 are the pipeline outputs for the model training, xt for test\n', color='red', attrs=['bold', 'dark']))
print(colored(f'\nSubmission sample file:-\n', color='blue', attrs=['bold', 'dark']))

mdl_pred_prf = pd.DataFrame(data=None, index=sub_fl.Id, columns=None)

def Train_Ensembles(mthd: str, n_estimators: np.int16=500, nb_mdl: np.int16=500):
    """
    This function implements the below routine- 
    1. Sample the dataset into train and test components based on random seed
    2. Invoke the model method
    3. Train the model on the train-set
    4. Accumulate test-set predictions and collate in output table
    
    Inputs- 
    1. mthd- (string):-  model method 
    2. n_estimators (int):- number of trees
    3. nb_mdl (int):- number of candidates 
    """
    global train_frac
    for mdl_nb in tqdm(range(nb_mdl), desc=f'\nModel training progress\n'):
        xtr = x1.sample(frac=train_frac, random_state=mdl_nb)
        ytr = y1.loc[xtr.index]
        (xdev, ydev) = (x1.loc[~x1.index.isin(xtr.index)], y1.loc[~y1.index.isin(ytr.index)])
        if mthd == 'LGBM':
            mdl = LGBMRegressor(n_estimators=n_estimators, max_depth=7, n_jobs=-1, learning_rate=0.08, objective='regression', metric=['rmse'])
        elif mthd == 'XgBoost':
            mdl = XGBRegressor(n_estimators=n_estimators, max_depth=9, n_jobs=-1, learning_rate=0.08, eval_metric=['rmse'])
        elif mthd == 'CatBoost':
            mdl = CatBoostRegressor(learning_rate=0.08, max_depth=9, eval_metric='RMSE')