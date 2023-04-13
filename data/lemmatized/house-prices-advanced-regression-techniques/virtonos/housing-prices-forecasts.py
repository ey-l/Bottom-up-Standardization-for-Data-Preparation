import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso, SGDRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import skew, skewtest, norm
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(_input1['SalePrice'], fit=norm)
print()
print('Skew is: %f' % _input1['SalePrice'].skew())
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'], c='blue', marker='s')
plt.title('Looking for outliers')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
_input1 = _input1[_input1['GrLivArea'] < 4500]
_input1 = _input1[_input1['SaleCondition'] == 'Normal']
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'], c='blue', marker='s')
plt.title('Looking for outliers')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
corrmatrix = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=0.8, square=True)
k = 10
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
Neighborhood = _input1.groupby('Neighborhood')
Neighborhood['SalePrice'].median()
ID_train = _input1['Id']
ID_test = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1 = _input1.drop('TotRmsAbvGrd', axis=1, inplace=False)
_input0 = _input0.drop('TotRmsAbvGrd', axis=1, inplace=False)
_input1 = _input1.drop('GarageYrBlt', axis=1, inplace=False)
_input0 = _input0.drop('GarageYrBlt', axis=1, inplace=False)
_input1 = _input1.drop('GarageArea', axis=1, inplace=False)
_input0 = _input0.drop('GarageArea', axis=1, inplace=False)
_input1 = _input1.drop('1stFlrSF', axis=1, inplace=False)
_input0 = _input0.drop('1stFlrSF', axis=1, inplace=False)
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
y = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1, inplace=False)
print(_input1.shape)
print(_input0.shape)
ntrain = _input1.shape[0]
ntest = _input0.shape[0]
print(ntrain)
Combined_data = pd.concat([_input1, _input0]).reset_index(drop=True)
print('Combined size is : {}'.format(Combined_data.shape))
total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum() / Combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
Combined_data = Combined_data.drop('PoolQC', axis=1, inplace=False)
Combined_data = Combined_data.drop('MiscFeature', axis=1, inplace=False)
Combined_data = Combined_data.drop('Alley', axis=1, inplace=False)
Combined_data['LotFrontage'] = Combined_data['LotFrontage'].fillna(0, inplace=False)
Combined_data['Fence'] = Combined_data['Fence'].fillna('None', inplace=False)
Combined_data['FireplaceQu'] = Combined_data['FireplaceQu'].fillna('None', inplace=False)
Combined_data['GarageCond'] = Combined_data['GarageCond'].fillna('None', inplace=False)
Combined_data['GarageFinish'] = Combined_data['GarageFinish'].fillna('None', inplace=False)
Combined_data['GarageQual'] = Combined_data['GarageQual'].fillna('None', inplace=False)
Combined_data['GarageType'] = Combined_data['GarageType'].fillna('None', inplace=False)
Combined_data['BsmtFinType2'] = Combined_data['BsmtFinType2'].fillna('None', inplace=False)
Combined_data['BsmtExposure'] = Combined_data['BsmtExposure'].fillna('None', inplace=False)
Combined_data['BsmtQual'] = Combined_data['BsmtQual'].fillna('None', inplace=False)
Combined_data['BsmtFinType1'] = Combined_data['BsmtFinType1'].fillna('None', inplace=False)
Combined_data['BsmtCond'] = Combined_data['BsmtCond'].fillna('None', inplace=False)
Combined_data['MasVnrType'] = Combined_data['MasVnrType'].fillna('None', inplace=False)
Combined_data['MasVnrArea'] = Combined_data['MasVnrArea'].fillna(0, inplace=False)
Combined_data['Electrical'] = Combined_data['Electrical'].fillna('SBrkr', inplace=False)
Combined_data['BsmtHalfBath'] = Combined_data['BsmtHalfBath'].fillna(0, inplace=False)
Combined_data['BsmtFullBath'] = Combined_data['BsmtFullBath'].fillna(0, inplace=False)
Combined_data['BsmtFinSF1'] = Combined_data['BsmtFinSF1'].fillna(0, inplace=False)
Combined_data['BsmtFinSF2'] = Combined_data['BsmtFinSF2'].fillna(0, inplace=False)
Combined_data['BsmtUnfSF'] = Combined_data['BsmtUnfSF'].fillna(0, inplace=False)
Combined_data['TotalBsmtSF'] = Combined_data['TotalBsmtSF'].fillna(0, inplace=False)
Combined_data['GarageCars'] = Combined_data['GarageCars'].fillna(0, inplace=False)
Combined_data['Utilities'] = Combined_data['Utilities'].fillna(0, inplace=False)
Combined_data['Functional'] = Combined_data['Functional'].fillna(0, inplace=False)
Combined_data['KitchenQual'] = Combined_data['KitchenQual'].fillna(0, inplace=False)
Combined_data['MSZoning'] = Combined_data['MSZoning'].fillna('RL', inplace=False)
Combined_data['SaleType'] = Combined_data['SaleType'].fillna('WD', inplace=False)
Combined_data['Exterior1st'] = Combined_data['Exterior1st'].fillna(Combined_data['Exterior1st'].mode()[0])
Combined_data['Exterior2nd'] = Combined_data['Exterior2nd'].fillna(Combined_data['Exterior2nd'].mode()[0])
Combined_data = Combined_data.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})
Combined_data = Combined_data.replace({'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'None': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, 'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}, 'Street': {'Grvl': 1, 'Pave': 2}, 'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}})
Combined_data['Total_Home_Quality'] = Combined_data['OverallQual'] + Combined_data['OverallCond']
Combined_data['Total_Basement_Quality'] = Combined_data['BsmtQual'] + Combined_data['BsmtCond']
Combined_data['Total_Basement_FinshedSqFt'] = Combined_data['BsmtFinSF1'] + Combined_data['BsmtFinSF2']
Combined_data['Total_Exterior_Quality'] = Combined_data['ExterQual'] + Combined_data['ExterCond']
Combined_data['Total_Garage_Quality'] = Combined_data['GarageCond'] + Combined_data['GarageQual'] + Combined_data['GarageFinish']
Combined_data['Total_Basement_FinshType'] = Combined_data['BsmtFinType1'] + Combined_data['BsmtFinType2']
Combined_data['Total_Garage_Quality'] = Combined_data['GarageCond'] + Combined_data['GarageQual'] + Combined_data['GarageFinish']
Combined_data['Total_Basement_FinshType'] = Combined_data['BsmtFinType1'] + Combined_data['BsmtFinType2']
Combined_data['Total_Bathrooms'] = Combined_data['BsmtFullBath'] + Combined_data['BsmtHalfBath'] * 0.5 + Combined_data['FullBath'] + Combined_data['HalfBath'] * 0.5
Combined_data['Total_Land_Quality'] = Combined_data['LandSlope'] + Combined_data['LotShape']
Combined_data = Combined_data.drop('OverallQual', axis=1, inplace=False)
Combined_data = Combined_data.drop('OverallCond', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtQual', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtCond', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtFinSF1', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtFinSF2', axis=1, inplace=False)
Combined_data = Combined_data.drop('ExterQual', axis=1, inplace=False)
Combined_data = Combined_data.drop('ExterCond', axis=1, inplace=False)
Combined_data = Combined_data.drop('GarageCond', axis=1, inplace=False)
Combined_data = Combined_data.drop('GarageQual', axis=1, inplace=False)
Combined_data = Combined_data.drop('GarageFinish', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtFinType1', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtFinType2', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtFullBath', axis=1, inplace=False)
Combined_data = Combined_data.drop('BsmtHalfBath', axis=1, inplace=False)
Combined_data = Combined_data.drop('FullBath', axis=1, inplace=False)
Combined_data = Combined_data.drop('HalfBath', axis=1, inplace=False)
Combined_data = Combined_data.drop('LandSlope', axis=1, inplace=False)
Combined_data = Combined_data.drop('LotShape', axis=1, inplace=False)
Combined_data = Combined_data.drop('LandContour', axis=1, inplace=False)
Combined_data = Combined_data.replace({'Neighborhood': {'MeadowV': 0, 'IDOTRR': 0, 'BrDale': 0, 'OldTown': 0, 'Edwards': 0, 'BrkSide': 0, 'Sawyer': 0, 'Blueste': 1, 'SWISU': 1, 'NAmes': 1, 'NPkVill': 1, 'Mitchel': 1, 'SawyerW': 1, 'Gilbert': 2, 'NWAmes': 2, 'Blmngtn': 2, 'CollgCr': 2, 'ClearCr': 2, 'Crawfor': 2, 'Veenker': 3, 'Somerst': 3, 'Timber': 3, 'StoneBr': 3, 'NoRidge': 3, 'NridgHt': 3}})
new_total = Combined_data.isnull().sum().sort_values(ascending=False)
new_percent = (Combined_data.isnull().sum() / Combined_data.isnull().count()).sort_values(ascending=False)
new_missing_data = pd.concat([new_total, new_percent], axis=1, keys=['Total', 'Percent'])
new_missing_data.head(10)
Skewed_Feature_Check = ['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'PoolArea', 'MiscVal', 'Total_Basement_FinshedSqFt']
for feature in Skewed_Feature_Check:
    print(feature, skew(Combined_data[feature]), skewtest(Combined_data[feature]))
    from scipy.special import boxcox1p
    lam = 0.15
    Combined_data[feature] = boxcox1p(Combined_data[feature], lam)
categorical_features = Combined_data.select_dtypes(include=['object']).columns
numerical_features = Combined_data.select_dtypes(exclude=['object']).columns
print('Numerical features : ' + str(len(numerical_features)))
print('Categorical features : ' + str(len(categorical_features)))
Combined_data_numerical = Combined_data[numerical_features]
Combined_data_categorical = Combined_data[categorical_features]
corrmatrix_combined = Combined_data_numerical.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix_combined, vmax=0.8, square=True)
Combined_data['FireplaceQu'].hist()
Combined_data['Fireplaces'].hist()
Combined_data = Combined_data.drop('FireplaceQu', axis=1, inplace=False)
Combined_data_categorical = pd.get_dummies(Combined_data_categorical, drop_first=True)
Combined_data = pd.concat([Combined_data_categorical, Combined_data_numerical], axis=1)
print('Combined size is : {}'.format(Combined_data.shape))
_input1 = Combined_data[:ntrain]
_input0 = Combined_data[ntrain:]
_input0 = _input0.reset_index(drop=True)
print(_input1.shape)
print(_input0.shape)
(X_train, X_test, y_train, y_test) = train_test_split(_input1, y, test_size=0.2, random_state=1)
print('X_train : ' + str(X_train.shape))
print('X_test : ' + str(X_test.shape))
print('y_train : ' + str(y_train.shape))
print('y_test : ' + str(y_test.shape))
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
for Model in [LinearRegression, Ridge, Lasso, XGBRegressor]:
    model = Model()
    print('%s: %s' % (Model.__name__, np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)).mean()))
alphas = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50]
plt.figure(figsize=(5, 3))
for model in [Lasso, Ridge]:
    scores = [np.sqrt(-cross_val_score(model(alpha), X_train, y_train, scoring='neg_mean_squared_error', cv=10)).mean() for alpha in alphas]
    plt.plot(alphas, scores, label=model.__name__)
    plt.legend(loc='center')
    plt.xlabel('alpha')
    plt.ylabel('cross validation score')
    plt.tight_layout()
xgbreg = XGBRegressor(nthreads=-1, booster='gblinear')
np.sqrt(-cross_val_score(xgbreg, X_train, y_train, scoring='neg_mean_squared_error', cv=10)).mean()
params = {'learning_rate': [1, 0.01], 'reg_alpha': [0.09], 'n_estimators': [1000]}
xgbreg = XGBRegressor(nthreads=-1, booster='gblinear')
from sklearn.model_selection import GridSearchCV
xgbreg_model = GridSearchCV(xgbreg, params, n_jobs=1, scoring='neg_mean_squared_error', cv=10)