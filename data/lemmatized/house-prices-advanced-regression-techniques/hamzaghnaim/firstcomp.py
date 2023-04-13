import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_copy = _input1.drop(['Id'], axis=1).copy()
test_copy = _input0.drop(['Id'], axis=1).copy()
nullColumns = [column for column in train_copy.columns if abs(train_copy[column].isnull().sum() / 1460 * 100) >= 40]
nullColumns
train_copy = train_copy.drop(columns=nullColumns, axis=1)
test_copy = test_copy.drop(columns=nullColumns, axis=1)

def impute(df):
    """
    This function imputes missing values for numeric and categorical datatypes
    """
    for name in df.select_dtypes(include=['int64', 'float64']):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes('object'):
        df[name] = df[name].fillna('None')
    return df
train_copy = impute(train_copy)
test_copy = impute(test_copy)
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

def score(df, y_transformed=False):
    """
    a function that measures the RMSE with each feature egireeing step
    """
    df = impute(df)
    X = df.copy()
    y = X.pop('SalePrice')
    if y_transformed == False:
        log_y = np.log1p(y)
    else:
        log_y = y
    for colname in X.select_dtypes(['object']):
        X[colname] = X[colname].astype('category')
        X[colname] = X[colname].cat.codes
    model = XGBRegressor()
    score = cross_val_score(model, X, log_y, cv=5, scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    print('Baseline score is : ', score)
score(train_copy)
train_copy.select_dtypes(include=['int64', 'float64']).var()
import matplotlib.pyplot as plt
import seaborn as sns
corr = train_copy.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(corr, annot=True)

def correlation_calculate(dataframe):
    correlated_features = set()
    coor_matrix = dataframe.corr()
    for x in range(len(coor_matrix.columns)):
        for y in range(x):
            if abs(coor_matrix.iloc[x, y]) > 0.7:
                clname = coor_matrix.columns[x]
                correlated_features.add(clname)
            else:
                pass
    return correlated_features
correlation_calculate(train_copy)
score(train_copy.drop(['TotalBsmtSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1))
import scipy.stats as stats

def chi_square_test(dataframe):
    """
    this function calculates the Cramer's V-test for categorical variables 
    if there is a dependence between them
        """
    for column1 in dataframe.select_dtypes('object'):
        for column2 in dataframe.select_dtypes('object'):
            myCrosstable = pd.crosstab(index=dataframe[column1], columns=dataframe[column2])
            (chiVal, pVal, df, exp) = stats.chi2_contingency(myCrosstable)
            if pVal < 0.05:
                if exp.min() >= 1 and len(exp[exp < 5]) / len(exp) * 100 <= 20:
                    n = np.sum(np.array(myCrosstable))
                    minDim = min(myCrosstable.shape) - 1
                    association_strength = np.sqrt(chiVal / n / minDim)
                    print('The association strength between {myField1} and {myField2} is : '.format(myField1=column1, myField2=column2), association_strength)
                else:
                    pass
            else:
                pass
chi_square_test(train_copy)
from sklearn.feature_selection import mutual_info_regression

def mutual_info_calculator(df):
    """
    calculate the mutual information between each feature and  the target variable
    """
    X = df.copy()
    for colname in X.select_dtypes('object'):
        (X[colname], _) = X[colname].factorize()
    discrete_features = X.dtypes == int
    mi_scores = mutual_info_regression(X, X['SalePrice'], discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=True)
    for i in range(len(mi_scores)):
        print(mi_scores.index[i], mi_scores[i])
mutual_info_calculator(train_copy)
train_copy = train_copy.drop(columns=['Utilities', 'MiscVal', 'MoSold', 'Street', 'YrSold', 'PoolArea'], inplace=False)
test_copy = test_copy.drop(columns=['Utilities', 'MiscVal', 'MoSold', 'Street', 'YrSold', 'PoolArea'], inplace=False)
score(train_copy)
import seaborn as sns
sns.distplot(train_copy['SalePrice'])
train_copy.select_dtypes(include=['int64', 'float64']).skew()
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
edit_feature = {'MSSubClass': {20: '1-STORY 1946 & NEWER ALL STYLES', 30: '1-STORY 1945 & OLDER', 40: '1-STORY W/FINISHED ATTIC ALL AGES', 45: '1-1/2 STORY - UNFINISHED ALL AGES', 50: '1-1/2 STORY FINISHED ALL AGES', 60: '2-STORY 1946 & NEWER', 70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES', 80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES', 120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER', 150: '1-1/2 STORY PUD - ALL AGES', 160: '2-STORY PUD - 1946 & NEWER', 180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', 190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'}}
train_copy = train_copy.replace(edit_feature)
test_copy = test_copy.replace(edit_feature)

def view_skew(dataframe):
    """
    this function determies which transformation is better
    by comparing the value of skew for each variable after each transformation

    """
    skewed_features = [column for column in dataframe.select_dtypes(include=['int64', 'float64'])]
    log = 0
    squareroot = 0
    recprocal = 0
    exponential = 0
    log_cols = []
    squareroot_cols = []
    recprocal_cols = []
    exponential_cols = []
    for column in skewed_features:
        log_transform = abs(np.log1p(dataframe[column]).skew())
        SquareRoot_transform = abs(np.sqrt(dataframe[column]).skew())
        reciprocal_transform = abs((1 / (dataframe[column] + 1)).skew())
        exponential_transform = abs((dataframe[column] ** (1 / 5)).skew())
        transforms = {'log': log_transform, 'squareroot': SquareRoot_transform, 'recprocal': reciprocal_transform, 'exponential': exponential_transform}
        best_score = [log_transform]
        the_transform = ['log']
        for (key, element) in transforms.items():
            if element < best_score[0]:
                best_score = [element]
                the_transform = [key]
            else:
                pass
        if the_transform[0] == 'log':
            log += 1
            log_cols.append(column)
        elif the_transform[0] == 'squareroot':
            squareroot += 1
            squareroot_cols.append(column)
        elif the_transform[0] == 'recprocal':
            recprocal += 1
            recprocal_cols.append(column)
        else:
            exponential += 1
            exponential_cols.append(column)
    print('count of features where log performed better :', log)
    print('count of features where square root performed better :', squareroot)
    print('count of features where reciprocal performed better :', recprocal)
    print('count of features where exponential performed better :', exponential)
    return (log_cols, squareroot_cols, recprocal_cols, exponential_cols)
(log_cols, squareroot_cols, recprocal_cols, exponential_cols) = view_skew(train_copy.copy())

def transform(df, features, transform):
    """
    This function applies the appropriate transform for each feature in order to fix skew
    """
    if transform == 'sqrt':
        df[features] = np.sqrt(df[features])
    elif transform == 'log':
        df[features] = np.log1p(df[features])
    elif transform == 'expo':
        df[features] = df[features] ** (1 / 5)
    else:
        df[features] = 1 / (df[features] + 1)
    return df
train_copy = transform(train_copy, log_cols, 'log')
train_copy = transform(train_copy, squareroot_cols, 'sqrt')
train_copy = transform(train_copy, recprocal_cols, 'recipro')
train_copy = transform(train_copy, exponential_cols, 'expo')
test_copy = transform(test_copy, exponential_cols, 'expo')
log_cols = ['LotArea', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'OpenPorchSF']
test_copy = transform(test_copy, log_cols, 'log')
test_copy = transform(test_copy, squareroot_cols, 'sqrt')
test_copy = transform(test_copy, recprocal_cols, 'recipro')
score(train_copy, True)
features = ['GarageCars', 'LotArea', 'TotalBsmtSF', 'YearBuilt', 'GrLivArea', 'GarageArea']
from sklearn.decomposition import PCA

def create_PC(df, features, standarize=True):
    """
    This function creates principle component features from variations in original features
    """
    X = df.loc[:, features]
    if standarize:
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    component_names = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X.columns)
    return (X_pca, loadings)
(X_pca, loadings) = create_PC(train_copy, features)
X_pca
loadings
train_copy['PCA_feature'] = train_copy.GarageCars * train_copy.GarageArea
train_copy = train_copy.join(X_pca)
(X_pca, loadings) = create_PC(test_copy, features)
test_copy['PCA_feature'] = test_copy.GarageCars * test_copy.GarageArea
test_copy = test_copy.join(X_pca)
score(train_copy, True)
nominal_features = ['MSSubClass', 'MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'SaleType', 'SaleCondition', 'Functional']
ordinal_features = [column for column in train_copy.select_dtypes(include=['object']).columns if column not in nominal_features]
for i in ordinal_features:
    print(i, train_copy[i].unique())
categories = {'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, 'None': 0}, 'LandContour': {'Lvl': 4, 'Bnk': 3, 'HLS': 2, 'Low': 1, 'None': 0}, 'LandSlope': {'Gtl': 3, 'Mod': 2, 'Sev': 1, 'None': 0}, 'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}, 'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, 'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, 'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}, 'PavedDrive': {'Y': 3, 'P': 2, 'N': 1, 'None': 0}, 'Electrical': {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1, 'None': 0}, 'GarageFinish': {'RFn': 3, 'Unf': 2, 'Fin': 1, 'None': 0}}

def ordinal_encode(df):
    """
    this function encodes ordinal categorical features
    """
    df = df.replace(categories)
    return df
train_copy = ordinal_encode(train_copy)
test_copy = ordinal_encode(test_copy)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')