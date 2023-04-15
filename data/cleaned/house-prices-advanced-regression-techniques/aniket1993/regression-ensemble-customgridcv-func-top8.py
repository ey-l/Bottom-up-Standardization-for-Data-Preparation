import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.describe().T
train.describe(include='O').T
sns.set_theme()
features = [i for i in train.iloc[:, 1:-1].columns if train[i].nunique() > 25]
plt.style.use(plt.style.available[19])
i = 1
plt.figure(figsize=(20, 25))
for j in features:
    plt.subplot(6, 3, i)
    sns.scatterplot(x=j, data=train, y=train['SalePrice'], hue='SaleCondition', palette='deep')
    plt.xlabel(j)
    i += 1

plt.figure(figsize=(22, 6))
sns.countplot(data=train, x='Neighborhood')
sns.set_theme()
j = 1
plt.figure(figsize=(24, 40))
for i in train['Neighborhood'].unique():
    plt.subplot(9, 3, j)
    sns.scatterplot(x=train[train['Neighborhood'] == i]['YearBuilt'], data=train, y=train[train['Neighborhood'] == i]['SalePrice'], hue='YrSold', palette='deep')
    plt.xlabel(i)
    j += 1
sns.set_theme()
fig = plt.figure(figsize=(20, 40))
for i in range(len(train.select_dtypes(include='object').columns)):
    fig.add_subplot(11, 4, i + 1)
    train.select_dtypes(include='object').iloc[:, i].value_counts().plot(kind='pie', subplots=True)
sns.set_theme()
for i in train[['BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea', 'LotArea', 'LotFrontage']].columns:
    plt.figure(figsize=(15, 6))
    sns.scatterplot(x=i, data=train, y=train['SalePrice'])
    plt.title('SalePrice against {}'.format(i))

train = train[~((train['BsmtFinSF2'] > 1200) | (train['ScreenPorch'] > 350) | (train['GrLivArea'] > 4000) | (train['OpenPorchSF'] > 350) | (train['EnclosedPorch'] > 350) | (train['BsmtFinSF1'] > 3000) | (train['MasVnrArea'] > 1200) | (train['LotArea'] > 100000) | (train['LotFrontage'] > 200))]
data = pd.concat([train, test], axis=0)
df = pd.DataFrame({'Type': data.dtypes, 'Missing': data.isna().sum(), 'Size': data.shape[0], 'Unique': data.nunique()})
df['Missing_%'] = df.Missing / df.Size * 100
df[df['Missing'] > 0].sort_values(by=['Missing_%'], ascending=False)
data['PoolQC'] = data['PoolQC'].fillna('NA')
data['MiscFeature'] = data['MiscFeature'].fillna('NA')
data['Alley'] = data['Alley'].fillna('NA')
data['Fence'] = data['Fence'].fillna('NA')
data['FireplaceQu'] = data['FireplaceQu'].fillna('NA')
data['GarageType'] = data['GarageType'].fillna('NA')
data['GarageFinish'] = data['GarageFinish'].fillna('NA')
data['BsmtCond'] = data['BsmtCond'].fillna('NA')
data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')
data['BsmtQual'] = data['BsmtQual'].fillna('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['GarageCond'] = data['GarageCond'].fillna('NA')
data['GarageQual'] = data['GarageQual'].fillna('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['MSZoning'] = data['MSZoning'].fillna('RL')
data['Functional'] = data['Functional'].fillna('Typ')
data['Utilities'] = data['Utilities'].fillna('AllPub')
data['KitchenQual'] = data['KitchenQual'].fillna('TA')
data['Exterior2nd'] = data['Exterior2nd'].fillna('VinylSd')
data['Exterior1st'] = data['Exterior1st'].fillna('VinylSd')
data['SaleType'] = data['SaleType'].fillna('WD')
data['MSSubClass'] = data['MSSubClass'].astype(object)
data['MoSold'] = data['MoSold'].astype(object)
df = pd.DataFrame({'Type': data.dtypes, 'Missing': data.isna().sum(), 'Size': data.shape[0], 'Unique': data.nunique()})
df['Missing_%'] = df.Missing / df.Size * 100
df[df['Missing'] > 0].sort_values(by=['Missing_%'], ascending=False)
for i in df[df['Missing'] > 0].index:
    if i == 'SalePrice':
        continue
    else:
        data[i] = data[i].fillna(data[i].median())
df = pd.DataFrame({'Type': data.dtypes, 'Missing': data.isna().sum(), 'Size': data.shape[0], 'Unique': data.nunique()})
df['Missing_%'] = df.Missing / df.Size * 100
df[df['Missing'] > 0].sort_values(by=['Missing_%'], ascending=False)
categorical = ['MSSubClass', 'MSZoning', 'LotConfig', 'Neighborhood', 'LandSlope', 'LandContour', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'Fence', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'PavedDrive', 'Alley', 'SaleCondition']
ordinal = ['LotShape', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond']
numerical = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal', 'GarageYrBlt']
ex_qu = {'Po': 0, 'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
ex_cond = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
Bsmt_Qual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
BsmtFinType1 = {'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
Bsmt_Exposure = {'NA': 0, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
garage_fin = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
garage_qu = {'NA': 0, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
LotShape = {'IR3': 0, 'IR2': 0, 'IR1': 1, 'Reg': 2}
Functional = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
data = data.replace({'LotShape': LotShape, 'ExterQual': ex_qu, 'ExterCond': ex_cond, 'BsmtQual': Bsmt_Qual, 'BsmtCond': Bsmt_Qual, 'BsmtExposure': Bsmt_Exposure, 'BsmtFinType1': BsmtFinType1, 'BsmtFinType2': BsmtFinType1, 'HeatingQC': ex_qu, 'KitchenQual': ex_qu, 'Functional': Functional, 'GarageFinish': garage_fin, 'GarageQual': garage_qu, 'GarageCond': garage_qu, 'FireplaceQu': garage_qu})
X1 = data[ordinal]
X2 = pd.get_dummies(data[categorical], drop_first=True)
X3 = data[numerical]
X3_train = X3.iloc[:len(train), :]
X3_test = X3.iloc[len(train):, :]
skewed_columns = []
for i in X3_train.columns:
    if abs(X3_train[i].skew()) > 0.5:
        skewed_columns.append(i)
from scipy.special import boxcox1p
lam = 0.15
for i in skewed_columns:
    X3_train[i] = boxcox1p(X3_train[i], lam)
from scipy.special import boxcox1p
lam = 0.15
for i in skewed_columns:
    X3_test[i] = boxcox1p(X3_test[i], lam)
X3 = pd.concat([X3_train, X3_test], axis=0)
dataset = pd.concat([X2, X1, X3], axis=1)
X = dataset.iloc[:len(train), :].values
Y = train.iloc[:, -1:].values
Y = np.log1p(Y)
test_dataset = dataset.iloc[len(train):, :]
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=1)
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train[:, len(X1.columns) + len(X2.columns):] = sc.fit_transform(X_train[:, len(X1.columns) + len(X2.columns):])
X_test[:, len(X1.columns) + len(X2.columns):] = sc.transform(X_test[:, len(X1.columns) + len(X2.columns):])
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import Lasso
reg = Lasso(alpha=0.0008)