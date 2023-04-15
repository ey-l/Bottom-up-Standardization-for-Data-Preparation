import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train
print('Shape of train data: ', df_train.shape)
print('Shape of test data: ', df_test.shape)
null_columns = df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], inplace=True)
df_test.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], inplace=True)
X = df_train.drop(columns=['SalePrice'])
y = df_train['SalePrice']
df = pd.concat([X, df_test], axis=0)
df
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
df[null_columns].dtypes
cleaner = KNNImputer(n_neighbors=11, weights='distance')
numerical = df[null_columns].select_dtypes(exclude='object').columns
df[numerical] = cleaner.fit_transform(df[numerical])
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
categorical = df[null_columns].select_dtypes(include='object').columns
cleaner = ColumnTransformer([('categorical_transformer', SimpleImputer(strategy='most_frequent'), categorical)])
df[null_columns] = cleaner.fit_transform(df[null_columns])
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
categorical = df.select_dtypes(include='object').columns
categorical
for i in range(0, len(categorical)):
    print(df[categorical[i]].value_counts())
    print('****************************************\n')
LotShape = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
df['LotShape'] = df['LotShape'].replace(LotShape)
MSZoning = {'C': 0, 'RH': 1, 'RM': 2, 'RL': 3, 'RP': 4}
df['MSZoning'] = df['MSZoning'].replace(MSZoning)
LandContour = {'Lvl': 0, 'HLS': 1, 'Bnk': 2, 'Low': 3}
df['LandContour'] = df['LandContour'].replace(LandContour)
LotConfig = {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4}
df['LotConfig'] = df['LotConfig'].replace(LotConfig)
LandSlope = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
df['LandSlope'] = df['LandSlope'].replace(LandSlope)
HouseStyle = {'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7}
df['HouseStyle'] = df['HouseStyle'].replace(HouseStyle)
ExterQual = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['ExterQual'] = df['ExterQual'].replace(ExterQual)
BsmtQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtQual'] = df['BsmtQual'].replace(BsmtQual)
BsmtCond = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['BsmtCond'] = df['BsmtCond'].replace(BsmtCond)
BsmtExposure = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Ex': 4}
df['BsmtExposure'] = df['BsmtExposure'].replace(BsmtExposure)
BsmtFinType1 = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].replace(BsmtFinType1)
BsmtFinType2 = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
df['BsmtFinType2'] = df['BsmtFinType2'].replace(BsmtFinType2)
Electrical = {'Mix': 0, 'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'SBrkr': 4}
df['Electrical'] = df['Electrical'].replace(Electrical)
KitchenQual = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
df['KitchenQual'] = df['KitchenQual'].replace(KitchenQual)
Functional = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
df['Functional'] = df['Functional'].replace(Functional)
GarageFinish = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
df['GarageFinish'] = df['GarageFinish'].replace(GarageFinish)
GarageQual = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageQual'] = df['GarageQual'].replace(GarageQual)
GarageCond = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['GarageCond'] = df['GarageCond'].replace(GarageCond)
PavedDrive = {'Dirt/Gravel': 0, 'Partial Pavement': 1, 'Paved': 2}
df['PavedDrive'] = df['PavedDrive'].replace(PavedDrive)
cat_col = df.select_dtypes(include=['object']).columns
encoder = preprocessing.LabelEncoder()
for i in range(0, len(cat_col)):
    df[cat_col[i]] = encoder.fit_transform(df[cat_col[i]].astype(str))
X = df.iloc[:1460, :]
df_test = df.iloc[1460:, :]
new_train = pd.concat([X, y], axis=1)
new_train
sns.boxplot(y=new_train['SalePrice'])
sns.distplot(new_train['SalePrice'])
new_train.drop(new_train[new_train['SalePrice'] > 500000].index, inplace=True)
sns.distplot(new_train['SalePrice'])
sns.boxplot(new_train['GrLivArea'])
sns.distplot(new_train['GrLivArea'])
new_train.drop(new_train[new_train['GrLivArea'] > 3000].index, inplace=True)
sns.distplot(new_train['GrLivArea'])
new_train
X = new_train.drop(columns=['SalePrice'])
y = new_train['SalePrice']
FeatureSelection = SelectPercentile(score_func=chi2, percentile=60)
X = FeatureSelection.fit_transform(X, y)
print('X Shape is ', X.shape)
corrmat = new_train.corr()
(f, ax) = plt.subplots(figsize=(30, 25))
cols = corrmat.nlargest(44, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(new_train[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

data = new_train[cols]
data
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, shuffle=True)
GBRModel = GradientBoostingRegressor(n_estimators=350, max_depth=3, learning_rate=0.1, random_state=44)