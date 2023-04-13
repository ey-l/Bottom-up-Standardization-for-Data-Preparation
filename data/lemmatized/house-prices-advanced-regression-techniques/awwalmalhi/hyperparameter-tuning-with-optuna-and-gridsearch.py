import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, train_test_split, KFold
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
pd.options.display.max_rows = 200
pd.set_option('mode.chained_assignment', None)
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter('ignore', category=ConvergenceWarning)
simplefilter('ignore', category=RuntimeWarning)
import optuna
from functools import partial
plt.style.use('fivethirtyeight')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
X_train = _input1.drop('SalePrice', axis=1)
y_train = np.log1p(_input1.SalePrice)
X_test = _input0

def missing_value_imputation(X):
    numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']
    ordinal_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
    categorical_features = [feature for feature in X.columns if feature not in ordinal_features and feature not in numerical_features]
    for feature in numerical_features:
        X[feature] = X[feature].fillna(X.groupby('Neighborhood')[feature].transform('median'))
    X['MSZoning'] = X['MSZoning'].replace({'RL': 'R', 'RM': 'R', 'RP': 'R', 'RM': 'R', 'I': 'O', 'A': 'O', 'C (all)': 'O', 'FV': 'O'}, inplace=False)
    X['Alley'] = X['Alley'].replace({np.nan: 'No', 'Grvl': 'Yes', 'Pave': 'Yes'}, inplace=False)
    X['LandContour'] = X['LandContour'].replace({'Lvl': 'Lvl', 'HLS': 'Slope', 'Bnk': 'Slope', 'Low': 'Slope'}, inplace=False)
    X['Condition1'] = X['Condition1'].where(X['Condition1'] == 'Norm', 'Other', inplace=False)
    X['HouseStyle'] = X['HouseStyle'].where((X['HouseStyle'] == '1Story') | (X['HouseStyle'] == '2Story') | (X['HouseStyle'] == '1.5Fin'), 'rare', inplace=False)
    X['RoofStyle'] = X['RoofStyle'].where((X['RoofStyle'] == 'Gable') | (X['RoofStyle'] == 'Hip'), 'rare', inplace=False)
    X['MasVnrType'] = X['MasVnrType'].where(X['MasVnrType'] == 'None', 'yes', inplace=False)
    X['MasVnrType'] = X['MasVnrType'].replace({np.nan: 'None'}, inplace=False)
    X['Exterior1st'] = X['Exterior1st'].where((X['Exterior1st'] == 'VinylSd') | (X['Exterior1st'] == 'HdBoard') | (X['Exterior1st'] == 'MetalSd') | (X['Exterior1st'] == 'Wd Sdng') | (X['Exterior1st'] == 'Plywood'), 'rare', inplace=False)
    X['PavedDrive'] = X['PavedDrive'].where(X['PavedDrive'] == 'Y', 'N', inplace=False)
    X['Fence'] = X['Fence'].replace({'MnPrv': 'Yes', 'GdPrv': 'Yes', 'GdWo': 'Yes', 'MnWw': 'Yes', np.nan: 'No'}, inplace=False)
    X['SaleType'] = X['SaleType'].where(X['SaleType'] == 'WD', 'other', inplace=False)
    X['SaleCondition'] = X['SaleCondition'].where((X['SaleCondition'] == 'Normal') | (X['SaleCondition'] == 'Partial') | (X['SaleCondition'] == 'Abnorml'), 'other', inplace=False)
    X['Neighborhood'] = X['Neighborhood'].replace({'Blmngtn': 'rare', 'BrDale': 'rare', 'Veenker': 'rare', 'NPkVill': 'rare', 'Blueste': 'rare', 'ClearCr': 'rare'}, inplace=False)
    for feature in categorical_features:
        val = X[feature].value_counts().index[0]
        X[feature] = X[feature].fillna(val)
    X['LotShape'] = X['LotShape'].replace({'Reg': 1, 'IR1': 0, 'IR2': 0, 'IR3': 0}, inplace=False)
    X['LandSlope'] = X['LandSlope'].replace({'Gtl': 3, 'Mod': 2, 'Sev': 1}, inplace=False)
    X['ExterQual'] = X['ExterQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['ExterCond'] = X['ExterCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['BsmtQual'] = X['BsmtQual'].replace({np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['BsmtCond'] = X['BsmtCond'].replace({np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['BsmtExposure'] = X['BsmtExposure'].replace({np.nan: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, inplace=False)
    X['BsmtFinType1'] = X['BsmtFinType1'].replace({np.nan: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=False)
    X['BsmtFinType2'] = X['BsmtFinType2'].replace({np.nan: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, inplace=False)
    X['HeatingQC'] = X['HeatingQC'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['KitchenQual'] = X['KitchenQual'].replace({np.nan: 3, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['Functional'] = X['Functional'].replace({np.nan: 8, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, inplace=False)
    X['FireplaceQu'] = X['FireplaceQu'].replace({np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['GarageFinish'] = X['GarageFinish'].replace({np.nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, inplace=False)
    X['GarageQual'] = X['GarageQual'].replace({np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['GarageCond'] = X['GarageCond'].replace({np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=False)
    X['PoolQC'] = X['PoolQC'].replace({np.nan: 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}, inplace=False)
    categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
    for feature in categorical_features:
        value_counts = X[feature].value_counts()
        for val in value_counts[value_counts / len(X) < 0.015].index:
            X[feature].loc[X[feature] == val] = 'rare'
    return X

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, combined_features=True, drop_underlying_features=True, drop_features=True, correlation_threshold=0.05, polynomial_features=True, outlier_threshold=1.5, empty_column_dropping_threshold=0.99, final_correlation_threshold=0.05, test=False):
        self.combined_features = combined_features
        self.drop_underlying_features = drop_underlying_features
        self.drop_features = drop_features
        self.correlation_threshold = correlation_threshold
        self.polynomial_features = polynomial_features
        self.polynomial_features_list = []
        self.outlier_threshold = outlier_threshold
        self.empty_column_dropping_threshold = empty_column_dropping_threshold
        self.low_correlation_drop_list = []
        self.one_hot_features_to_drop = []
        self.final_low_correlation_drop_list = []
        self.final_correlation_threshold = final_correlation_threshold
        self.test = _input0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ordinal_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O' and feature not in ordinal_features]
        categorical_features = [feature for feature in X.columns if feature not in ordinal_features and feature not in numerical_features]
        X['GarageYrBlt'] = 2020 - X['GarageYrBlt']
        X['YrSold'] = 2020 - X['YrSold']
        X['YearBuilt'] = 2020 - X['YearBuilt']
        X['YearRemodAdd'] = 2020 - X['YearRemodAdd']
        if self.combined_features:
            X['OverallGrade'] = X['OverallQual'] * X['OverallCond']
            X['ExterGrade'] = X['ExterQual'] * X['ExterCond']
            X['BsmtGrade'] = X['BsmtQual'] * X['BsmtCond']
            X['BsmtFinType'] = X['BsmtFinType1'] + X['BsmtFinType2']
            X['BsmtFinSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
            X['FlrSF'] = X['1stFlrSF'] + X['2ndFlrSF']
            X['BsmtBath'] = X['BsmtFullBath'] + X['BsmtHalfBath']
            X['Bath'] = X['FullBath'] + X['HalfBath']
            X['GarageGrade'] = X['GarageQual'] * X['GarageCond']
            X['Porch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
            numerical_features.extend(['OverallGrade', 'ExterGrade', 'BsmtGrade', 'BsmtFinType', 'BsmtFinSF', 'FlrSF', 'BsmtBath', 'Bath', 'GarageGrade', 'Porch'])
        if self.combined_features and self.drop_underlying_features:
            to_drop = ['BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', '1stFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'GarageCond', 'OpenPorchSF', 'BsmtFinSF2', '2ndFlrSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
            X = X.drop(to_drop, axis=1, inplace=False)
        if self.drop_features:
            to_drop = ['Utilities', 'PoolQC', 'MiscFeature', 'Street', 'Condition2', 'MasVnrType', 'LowQualFinSF', 'Alley', 'MiscVal', 'Fence', 'KitchenAbvGr', 'PoolArea', 'Street', 'RoofMatl', 'Exterior2nd', 'Heating']
            X = X.drop(to_drop, axis=1, inplace=False)
        ordinal_features = [feature for feature in X.columns if X[feature].dtype != 'O' and feature not in numerical_features]
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O' and feature not in ordinal_features]
        categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
        if self.polynomial_features:
            if len(X) > 800 and (not self.test):
                y = _input1['SalePrice'].iloc[X.index].to_frame()
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                self.polynomial_features_list = list(saleprice.sort_values(ascending=False).index[1:self.polynomial_features + 2])
                for feature in self.polynomial_features_list:
                    for i in [2, 3]:
                        X[f'{feature} - sq{i}'] = X[feature] ** i
            else:
                for feature in self.polynomial_features_list:
                    for i in [2, 3]:
                        X[f'{feature} - sq{i}'] = X[feature] ** i
        if self.outlier_threshold:
            numerical_features = [feature for feature in X.columns if feature not in ordinal_features and X[feature].dtype != 'O']
            for feature in numerical_features:
                unique_vals = X[feature].nunique()
                if unique_vals > 10:
                    q1 = np.percentile(X[feature], 25)
                    q3 = np.percentile(X[feature], 75)
                    iqr = q3 - q1
                    if not iqr:
                        iqr = 1
                    cut_off = iqr * self.outlier_threshold
                    (lower, upper) = (q1 - cut_off, q3 + cut_off)
                    X[feature] = X[feature].where(~(X[feature] > upper), upper, inplace=False)
                    X[feature] = X[feature].where(~(X[feature] < lower), lower, inplace=False)
        if self.correlation_threshold:
            if len(X) > 800 and (not self.test):
                self.low_correlation_drop_list = []
                y = _input1['SalePrice'].iloc[X.index].to_frame()
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                corr_dict = saleprice.sort_values(ascending=False).to_frame().to_dict()['SalePrice']
                for (key, value) in corr_dict.items():
                    if value < self.correlation_threshold and value > -self.correlation_threshold:
                        self.low_correlation_drop_list.append(key)
                X = X.drop(self.low_correlation_drop_list, axis=1)
            else:
                X = X.drop(self.low_correlation_drop_list, axis=1)
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']
        categorical_features = [feature for feature in X.columns if feature not in numerical_features]
        X = X.replace({0: 1e-05}, inplace=False)
        X_index = X.index
        num_pipeline = Pipeline([('scale', MinMaxScaler(feature_range=(1e-05, 1))), ('power_transform', PowerTransformer(method='box-cox'))])
        transformer = ColumnTransformer([('num_pipeline', num_pipeline, numerical_features)])
        numerical_features_transformed = transformer.fit_transform(X)
        numerical_df = pd.DataFrame(numerical_features_transformed, columns=numerical_features, index=X.index)
        X = pd.concat([numerical_df, X.loc[:, categorical_features]], axis=1)
        if not self.test:
            X_total = pd.concat([X, X_train_imputed.loc[:, categorical_features].loc[~X_train_imputed.index.isin(X.index)]])
            X_total = pd.get_dummies(X_total)
            one_hot_features = [feature for feature in X_total.columns if feature not in numerical_features]
        else:
            X = pd.get_dummies(X)
        if self.empty_column_dropping_threshold:
            if len(X) > 800 and (not self.test):
                self.one_hot_features_to_drop = []
                for feature in one_hot_features:
                    zero_count = X_total[feature].value_counts()[0] / len(X_total)
                    if zero_count > self.empty_column_dropping_threshold:
                        self.one_hot_features_to_drop.append(feature)
                        X_total = X_total.drop(feature, axis=1, inplace=False)
            elif not self.test:
                X_total = X_total.drop(self.one_hot_features_to_drop, axis=1, inplace=False)
            else:
                X = X.drop(self.one_hot_features_to_drop, axis=1, inplace=False)
        if not self.test:
            X = X_total.dropna()
        if self.final_correlation_threshold:
            if len(X) > 800 and (not self.test):
                self.final_low_correlation_drop_list = []
                y = _input1['SalePrice'].iloc[X.index].to_frame()
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                corr_dict = saleprice.sort_values(ascending=False).to_frame().to_dict()['SalePrice']
                for (key, value) in corr_dict.items():
                    if value < self.final_correlation_threshold and value > -self.final_correlation_threshold:
                        self.final_low_correlation_drop_list.append(key)
                X = X.drop(self.final_low_correlation_drop_list, axis=1)
            else:
                X = X.drop(self.final_low_correlation_drop_list, axis=1)
        return X
X_train_imputed = missing_value_imputation(X_train.copy())
param_grid = {'feature_engineering__combined_features': [True], 'feature_engineering__drop_features': [True], 'feature_engineering__drop_underlying_features': [True], 'feature_engineering__polynomial_features': [9], 'feature_engineering__correlation_threshold': [0.07], 'feature_engineering__outlier_threshold': [1.5], 'feature_engineering__empty_column_dropping_threshold': [0.99], 'feature_engineering__final_correlation_threshold': [False]}
linreg = Pipeline([('feature_engineering', FeatureEngineering()), ('linreg', Ridge())])
grid = GridSearchCV(linreg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)