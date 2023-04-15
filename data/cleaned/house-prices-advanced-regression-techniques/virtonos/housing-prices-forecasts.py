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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(train_data['SalePrice'], fit=norm)
print()
print('Skew is: %f' % train_data['SalePrice'].skew())
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c='blue', marker='s')
plt.title('Looking for outliers')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

train_data = train_data[train_data['GrLivArea'] < 4500]
train_data = train_data[train_data['SaleCondition'] == 'Normal']
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c='blue', marker='s')
plt.title('Looking for outliers')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

corrmatrix = train_data.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=0.8, square=True)
k = 10
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

Neighborhood = train_data.groupby('Neighborhood')
Neighborhood['SalePrice'].median()
ID_train = train_data['Id']
ID_test = test_data['Id']
train_data.drop('Id', axis=1, inplace=True)
test_data.drop('Id', axis=1, inplace=True)
train_data.drop('TotRmsAbvGrd', axis=1, inplace=True)
test_data.drop('TotRmsAbvGrd', axis=1, inplace=True)
train_data.drop('GarageYrBlt', axis=1, inplace=True)
test_data.drop('GarageYrBlt', axis=1, inplace=True)
train_data.drop('GarageArea', axis=1, inplace=True)
test_data.drop('GarageArea', axis=1, inplace=True)
train_data.drop('1stFlrSF', axis=1, inplace=True)
test_data.drop('1stFlrSF', axis=1, inplace=True)
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
y = train_data['SalePrice']
train_data.drop('SalePrice', axis=1, inplace=True)
print(train_data.shape)
print(test_data.shape)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
print(ntrain)
Combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)
print('Combined size is : {}'.format(Combined_data.shape))
total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum() / Combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
Combined_data.drop('PoolQC', axis=1, inplace=True)
Combined_data.drop('MiscFeature', axis=1, inplace=True)
Combined_data.drop('Alley', axis=1, inplace=True)
Combined_data['LotFrontage'].fillna(0, inplace=True)
Combined_data['Fence'].fillna('None', inplace=True)
Combined_data['FireplaceQu'].fillna('None', inplace=True)
Combined_data['GarageCond'].fillna('None', inplace=True)
Combined_data['GarageFinish'].fillna('None', inplace=True)
Combined_data['GarageQual'].fillna('None', inplace=True)
Combined_data['GarageType'].fillna('None', inplace=True)
Combined_data['BsmtFinType2'].fillna('None', inplace=True)
Combined_data['BsmtExposure'].fillna('None', inplace=True)
Combined_data['BsmtQual'].fillna('None', inplace=True)
Combined_data['BsmtFinType1'].fillna('None', inplace=True)
Combined_data['BsmtCond'].fillna('None', inplace=True)
Combined_data['MasVnrType'].fillna('None', inplace=True)
Combined_data['MasVnrArea'].fillna(0, inplace=True)
Combined_data['Electrical'].fillna('SBrkr', inplace=True)
Combined_data['BsmtHalfBath'].fillna(0, inplace=True)
Combined_data['BsmtFullBath'].fillna(0, inplace=True)
Combined_data['BsmtFinSF1'].fillna(0, inplace=True)
Combined_data['BsmtFinSF2'].fillna(0, inplace=True)
Combined_data['BsmtUnfSF'].fillna(0, inplace=True)
Combined_data['TotalBsmtSF'].fillna(0, inplace=True)
Combined_data['GarageCars'].fillna(0, inplace=True)
Combined_data['Utilities'].fillna(0, inplace=True)
Combined_data['Functional'].fillna(0, inplace=True)
Combined_data['KitchenQual'].fillna(0, inplace=True)
Combined_data['MSZoning'].fillna('RL', inplace=True)
Combined_data['SaleType'].fillna('WD', inplace=True)
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
Combined_data.drop('OverallQual', axis=1, inplace=True)
Combined_data.drop('OverallCond', axis=1, inplace=True)
Combined_data.drop('BsmtQual', axis=1, inplace=True)
Combined_data.drop('BsmtCond', axis=1, inplace=True)
Combined_data.drop('BsmtFinSF1', axis=1, inplace=True)
Combined_data.drop('BsmtFinSF2', axis=1, inplace=True)
Combined_data.drop('ExterQual', axis=1, inplace=True)
Combined_data.drop('ExterCond', axis=1, inplace=True)
Combined_data.drop('GarageCond', axis=1, inplace=True)
Combined_data.drop('GarageQual', axis=1, inplace=True)
Combined_data.drop('GarageFinish', axis=1, inplace=True)
Combined_data.drop('BsmtFinType1', axis=1, inplace=True)
Combined_data.drop('BsmtFinType2', axis=1, inplace=True)
Combined_data.drop('BsmtFullBath', axis=1, inplace=True)
Combined_data.drop('BsmtHalfBath', axis=1, inplace=True)
Combined_data.drop('FullBath', axis=1, inplace=True)
Combined_data.drop('HalfBath', axis=1, inplace=True)
Combined_data.drop('LandSlope', axis=1, inplace=True)
Combined_data.drop('LotShape', axis=1, inplace=True)
Combined_data.drop('LandContour', axis=1, inplace=True)
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
Combined_data.drop('FireplaceQu', axis=1, inplace=True)
Combined_data_categorical = pd.get_dummies(Combined_data_categorical, drop_first=True)
Combined_data = pd.concat([Combined_data_categorical, Combined_data_numerical], axis=1)
print('Combined size is : {}'.format(Combined_data.shape))
train_data = Combined_data[:ntrain]
test_data = Combined_data[ntrain:]
test_data = test_data.reset_index(drop=True)
print(train_data.shape)
print(test_data.shape)
(X_train, X_test, y_train, y_test) = train_test_split(train_data, y, test_size=0.2, random_state=1)
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