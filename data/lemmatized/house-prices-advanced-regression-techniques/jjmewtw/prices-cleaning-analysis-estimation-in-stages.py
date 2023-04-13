import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import variation
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import minmax_scaling
import math
from xgboost import XGBRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
RandState = 100
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.head()
_input0.head()
_input1['IsTrain'] = 1
_input0['IsTrain'] = 0
DataRaw = pd.concat([_input1, _input0])
DataRaw.head()
DataRaw.describe()
print('Number of features not counting the target:' + str(len(DataRaw.columns) - 1 - 1))
C = DataRaw.dtypes == 'object'
CategoricalVariables = list(C[C].index)
print(CategoricalVariables)
print('')
print('The number of categorical variables:' + str(len(CategoricalVariables)))
Integer = DataRaw.dtypes == 'int64'
Float = DataRaw.dtypes == 'float64'
NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)
print(NumericVariables)
print('')
print('The number of numeric variables:' + str(len(NumericVariables)))
Missing_Percentage = DataRaw.isnull().sum().sum() / np.product(DataRaw.shape) * 100
print('The number of missing entries: ' + str(round(Missing_Percentage, 2)) + ' %')
Numeric_NaN = DataRaw[NumericVariables].isnull().sum()
RowsCount = len(DataRaw.index)
print('The percentage number of missing entries per variable: ', format(round(Numeric_NaN / RowsCount * 100)))
CleanedNumeric = DataRaw[NumericVariables]
CleanedNumeric['GarageYrBlt'] = CleanedNumeric['GarageYrBlt'].fillna(CleanedNumeric['GarageYrBlt'].median())
CleanedNumeric.GarageYrBlt[CleanedNumeric.GarageYrBlt > 2020] = CleanedNumeric['GarageYrBlt'].median()
CleanedNumeric['LotFrontage'] = CleanedNumeric['LotFrontage'].fillna(CleanedNumeric['LotFrontage'].median())
CleanedNumeric = CleanedNumeric.fillna(0)
CleanedNumeric.head()
CleanedNumeric.describe()
CoefVar = pd.DataFrame(variation(CleanedNumeric), index=NumericVariables, columns=['CoefVar']).sort_values(by=['CoefVar'])
CoefVar
sns.distplot(a=CleanedNumeric['YrSold'], kde=False)

def PlotDist(NameOfVar):
    sns.distplot(a=CleanedNumeric[NameOfVar], kde=False)
sns.distplot(a=CleanedNumeric['PoolArea'], kde=False)
sns.distplot(a=CleanedNumeric['MiscVal'], kde=False)
sns.distplot(a=CleanedNumeric['LowQualFinSF'], kde=False)
sns.distplot(a=CleanedNumeric['3SsnPorch'], kde=False)
sns.distplot(a=CleanedNumeric['BsmtHalfBath'], kde=False)
Categorical_NaN = DataRaw[CategoricalVariables].isnull().sum()
RowsCount = len(DataRaw.index)
print('The percentage number of missing entries per variable: ', format(round(Categorical_NaN / RowsCount * 100)))
LuxuriousCategoricalVariables = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
CategoricalVariables = [x for x in CategoricalVariables if x not in LuxuriousCategoricalVariables]
print(CategoricalVariables)
print(LuxuriousCategoricalVariables)
CleanedCategorical = DataRaw[CategoricalVariables].fillna('Unknown')
LuxuriousCategorical = DataRaw[LuxuriousCategoricalVariables]
LuxuriousCategorical = pd.concat([LuxuriousCategorical, pd.DataFrame(DataRaw[LuxuriousCategoricalVariables].isnull().sum(axis=1), columns=['Luxurious_Features'])], axis=1, sort=False)
LuxuriousCategorical['Luxurious_Features'] = -LuxuriousCategorical['Luxurious_Features'] + 6
LuxuriousCategorical.head()
CleanedCategorical = pd.merge(CleanedCategorical, LuxuriousCategorical['Luxurious_Features'], on='Id')
CleanedCategorical.head()
CleanedCategorical['ExterQual'].unique()
Quality_map = {'NaN': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
CleanedCategorical['ExterQual'] = CleanedCategorical['ExterQual'].map(Quality_map)
CleanedCategorical['ExterCond'] = CleanedCategorical['ExterCond'].map(Quality_map)
CleanedCategorical['HeatingQC'] = CleanedCategorical['HeatingQC'].map(Quality_map)
CleanedCategorical['KitchenQual'] = CleanedCategorical['KitchenQual'].map(Quality_map)
Quality2_map = {'NaN': 1, 'NA': 1, 'Po': 2, 'Fa': 3, 'TA': 4, 'Gd': 5, 'Ex': 6}
CleanedCategorical['BsmtQual'] = CleanedCategorical['BsmtQual'].map(Quality2_map)
CleanedCategorical['BsmtCond'] = CleanedCategorical['BsmtCond'].map(Quality2_map)
CleanedCategorical['GarageQual'] = CleanedCategorical['GarageQual'].map(Quality2_map)
CleanedCategorical['GarageCond'] = CleanedCategorical['GarageCond'].map(Quality2_map)
Quality3_map = {'NaN': 1, 'NA': 1, 'No': 2, 'Mn': 3, 'Av': 4, 'Gd': 5}
CleanedCategorical['BsmtExposure'] = CleanedCategorical['BsmtExposure'].map(Quality3_map)
Quality4_map = {'NaN': 1, 'NA': 1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5, 'ALQ': 7, 'GLQ': 7}
CleanedCategorical['BsmtFinType1'] = CleanedCategorical['BsmtFinType1'].map(Quality4_map)
CleanedCategorical['BsmtFinType2'] = CleanedCategorical['BsmtFinType2'].map(Quality4_map)
Quality5_map = {'NaN': 1, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 3, 'Mod': 4, 'Min1': 5, 'Min2': 5, 'Typ': 6}
CleanedCategorical['Functional'] = CleanedCategorical['Functional'].map(Quality5_map)
Quality6_map = {'NaN': 1, 'NA': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4}
CleanedCategorical['GarageFinish'] = CleanedCategorical['GarageFinish'].map(Quality6_map)
OrdinalVariables = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish']
CleanedOrdinal = CleanedCategorical[OrdinalVariables]
CleanedOrdinal = pd.merge(CleanedOrdinal, LuxuriousCategorical['Luxurious_Features'], on='Id')
OrdinalVariables = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish', 'Luxurious_Features']
CleanedOrdinal = CleanedOrdinal[OrdinalVariables].fillna(1)
CleanedOrdinal.head()
print(CleanedOrdinal['BsmtQual'].loc[[18]])
NominalVariables = [x for x in CategoricalVariables if x not in OrdinalVariables]
AllLevelsPerVar = CleanedCategorical[NominalVariables].nunique()
AllLevels = CleanedCategorical[NominalVariables].nunique().sum()
print(AllLevelsPerVar)
print('Number of all levels coming from nominal variables: ' + str(AllLevels))
CleanedCategoricalDummy = pd.get_dummies(CleanedCategorical[NominalVariables], columns=NominalVariables)
CleanedCategoricalDummy.head()
CleanedTotal = pd.merge(CleanedNumeric, CleanedOrdinal, on='Id')
CleanedTotal = pd.merge(CleanedTotal, CleanedCategoricalDummy, on='Id')
CleanedTotal.head()
Target = ['IsTrain', 'SalePrice']
AllVariables = list(CleanedTotal.columns)
NumericVariablesNoTarget = [x for x in NumericVariables if x not in Target]
AllVariablesNoTarget = [x for x in AllVariables if x not in Target]
ScaledCleanedTotal = CleanedTotal
ScaledCleanedTotal[NumericVariablesNoTarget] = minmax_scaling(CleanedTotal, columns=NumericVariablesNoTarget)
ScaledCleanedTotal.head()
DataTrain = CleanedTotal[ScaledCleanedTotal.IsTrain == 1]
DataTest = CleanedTotal[ScaledCleanedTotal.IsTrain == 0]
sns.distplot(DataTrain['SalePrice'])
TrainTargetMean = DataTrain['SalePrice'].mean()
TrainTargetVar = DataTrain['SalePrice'].var()
TrainTargetSkew = DataTrain['SalePrice'].skew()
TrainTargetKurt = DataTrain['SalePrice'].kurt()
print('Mean: ' + str(round(TrainTargetMean)) + ' with std: ' + str(round(TrainTargetVar ** (1 / 2))) + ', skewness: ' + str(round(TrainTargetSkew, 1)) + ', and kurtosis: ' + str(round(TrainTargetKurt, 1)) + '.')
plt.figure(figsize=(18, 3))
plt.subplot(131)
plt.scatter('OverallQual', 'SalePrice', data=DataTrain)
plt.subplot(132)
plt.scatter('GrLivArea', 'SalePrice', data=DataTrain)
plt.subplot(133)
plt.scatter('YrSold', 'SalePrice', data=DataTrain)
CorrelationMatrix = DataTrain.corr()
(fig, axe) = plt.subplots(figsize=(15, 10))
sns.heatmap(CorrelationMatrix, vmax=0.9, square=True)
VarNo = 15
TopCorrelatedColumns = CorrelationMatrix.nlargest(VarNo, 'SalePrice')['SalePrice'].index
Reduced = np.corrcoef(DataTrain[TopCorrelatedColumns].values.T)
(fig, axe) = plt.subplots(figsize=(15, 10))
sns.heatmap(Reduced, vmax=0.9, square=True, yticklabels=TopCorrelatedColumns.values, xticklabels=TopCorrelatedColumns.values, annot=True, annot_kws={'size': 10})
selector_F = SelectKBest(f_classif, k=VarNo)
Selected_F = selector_F.fit_transform(DataTrain[AllVariablesNoTarget], DataTrain['SalePrice'])
SelectedOrdered_F = pd.DataFrame(selector_F.inverse_transform(Selected_F), index=DataTrain.index, columns=AllVariablesNoTarget)
SelectedOrdered_F.head()
SelectedVariables_F = list(SelectedOrdered_F.columns[SelectedOrdered_F.var() > 0])
DataTrain[SelectedVariables_F].head()
L1_par = 0.22