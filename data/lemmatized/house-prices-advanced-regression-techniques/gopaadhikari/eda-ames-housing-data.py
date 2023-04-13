import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from scipy import stats
import statsmodels.api as sm
from sklearn import preprocessing
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.Max_columns', 100)
pd.set_option('display.Max_rows', 100)
from sklearn.impute import SimpleImputer
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
label = 'SalePrice'
_input1 = _input1.drop(columns=['Id'], inplace=False)
print('Data shape:', _input1.shape)
print('------------------------\nData types:\n', _input1.dtypes.value_counts())
print('------------------------')
_input1.head(10)
for col in _input1:
    if col[0].isdigit():
        _input1 = _input1.rename(columns={col: 'n' + col}, inplace=False)
_input1['AgeBuilt'] = _input1['YrSold'] - _input1['YearBuilt']
_input1['AgeRemod'] = _input1['YrSold'] - _input1['YearRemodAdd']
_input1['GarageAge'] = _input1['YrSold'] - _input1['GarageYrBlt']
_input1 = _input1.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=False)

def map_ordinals(df):
    LotShape = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
    _input1.LotShape = _input1.LotShape.map(LotShape)
    LandContour = {'Low': 0, 'Lvl': 1, 'Bnk': 2, 'HLS': 3}
    _input1.LandContour = _input1.LandContour.map(LandContour)
    LandSlope = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
    _input1.LandSlope = _input1.LandSlope.map(LandSlope)
    ExterQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    _input1.ExterQual = _input1.ExterQual.map(ExterQual)
    ExterCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    _input1.ExterCond = _input1.ExterCond.map(ExterCond)
    BsmtQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    _input1.BsmtQual = _input1.BsmtQual.map(BsmtQual)
    BsmtCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    _input1.BsmtCond = _input1.BsmtCond.map(BsmtCond)
    BsmtExposure = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    _input1.BsmtExposure = _input1.BsmtExposure.map(BsmtExposure)
    BsmtFinType1 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    _input1.BsmtFinType1 = _input1.BsmtFinType1.map(BsmtFinType1)
    BsmtFinType2 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    _input1.BsmtFinType2 = _input1.BsmtFinType2.map(BsmtFinType2)
    HeatingQC = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    _input1.HeatingQC = _input1.HeatingQC.map(HeatingQC)
    KitchenQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    _input1.KitchenQual = _input1.KitchenQual.map(KitchenQual)
    Functional = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
    _input1.Functional = _input1.Functional.map(Functional)
    FireplaceQu = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    _input1.FireplaceQu = _input1.FireplaceQu.map(FireplaceQu)
    GarageFinish = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
    _input1.GarageFinish = _input1.GarageFinish.map(GarageFinish)
    GarageQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    _input1.GarageQual = _input1.GarageQual.map(GarageQual)
    GarageCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    _input1.GarageCond = _input1.GarageCond.map(GarageCond)
    PavedDrive = {'Y': 2, 'P': 1, 'N': 0}
    _input1.PavedDrive = _input1.PavedDrive.map(PavedDrive)
    PoolQC = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0}
    _input1.PoolQC = _input1.PoolQC.map(PoolQC)
    Fence = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
    _input1.Fence = _input1.Fence.map(Fence)
    return _input1
_input1 = map_ordinals(_input1)
print('Data shape:', _input1.shape)
print('------------------------\nData types:\n', _input1.dtypes.value_counts())
print('------------------------')
n_obs = _input1.shape[0]
missing_df = pd.DataFrame(columns=['Dtype', 'Missing', 'Missing ratio'])
for col in _input1:
    n_missing = _input1[col].isnull().sum()
    if n_missing:
        missing_df.loc[col] = [_input1[col].dtype, n_missing, n_missing / n_obs * 100]
missing_df.sort_values(by=['Dtype', 'Missing'], ascending=True)
imputer0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
_input1[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'GarageAge', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'PoolQC']] = imputer0.fit_transform(_input1[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'GarageAge', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'PoolQC']].values)
imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
_input1[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'KitchenQual', 'Functional']] = imputer1.fit_transform(_input1[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'KitchenQual', 'Functional']].values)
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
_input1[['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'SaleType']] = imputer2.fit_transform(_input1[['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'SaleType']].values)
_input1[['GarageType']] = _input1[['GarageType']].replace(np.nan, 'No garage')
_input1[['Alley']] = _input1[['Alley']].replace(np.nan, 'No alley')
_input1[['MiscFeature']] = _input1[['MiscFeature']].replace(np.nan, 'None')
_input1 = _input1.dropna(axis=1, inplace=False)
print('Data shape after missing value imputation: ', _input1.shape)
_input1

def num_univarstats_r(df, numeric=True):
    corr_col = 'Corr ' + label
    univar_df = pd.DataFrame(columns=['Dtype', 'Count', 'Unique', 'Mode', 'Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Std', 'Skew', 'Kurt', corr_col])
    for col in _input1:
        if pd.api.types.is_numeric_dtype(_input1[col]):
            univar_df.loc[col] = [_input1[col].dtype, _input1[col].count(), _input1[col].nunique(), _input1[col].mode().values[0], _input1[col].mean(), _input1[col].min(), _input1[col].quantile(0.25), _input1[col].median(), _input1[col].quantile(0.75), _input1[col].max(), _input1[col].std(), _input1[col].skew(), _input1[col].kurt(), _input1[col].corr(_input1[label])]
    return univar_df
df_stats = num_univarstats_r(_input1).sort_values(by=['Dtype', 'Corr SalePrice', 'Skew'], ascending=False)
df_stats
_input1[label] = np.log(_input1[label])
print('SalePrice Skewness:', _input1[label].skew(), ' Kurtosis:', _input1[label].kurt())
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121)
sns.histplot(data=_input1, x=label, kde=True, ax=ax)
ax = fig.add_subplot(122)
stats.probplot(_input1[label], plot=ax)
skwdlist = df_stats.index[(round(abs(df_stats['Corr SalePrice']), 1) > 0.1) & (abs(df_stats['Corr SalePrice']) != 1) & (round(abs(df_stats['Skew'])) > 1)].tolist()
print(skwdlist)

def remove_num_outlier(df, col):
    Q1 = np.percentile(_input1[col], 25, interpolation='midpoint')
    Q3 = np.percentile(_input1[col], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    print('Old Shape: ', _input1.shape)
    upper = np.where(_input1[col] >= Q3 + 1.5 * IQR)
    lower = np.where(_input1[col] <= Q1 - 1.5 * IQR)
    _input1 = _input1.drop(upper[0], inplace=False)
    _input1 = _input1.drop(lower[0], inplace=False)
    print('New Shape: ', _input1.shape)

def remove_grph_outlier(df, col, xlim, ylim=None):
    print(col, ' Skew before:', _input1[col].skew())
    if ylim:
        _input1 = _input1[~((_input1[col] > xlim) & (_input1[label] < ylim))]
    else:
        _input1 = _input1[_input1[col] < xlim]
    print(col, ' Skew after:', _input1[col].skew())
    print(_input1.shape)
    return _input1
_input1 = remove_grph_outlier(_input1, 'LotFrontage', 300)
_input1 = remove_grph_outlier(_input1, 'LotArea', 100000)
_input1 = remove_grph_outlier(_input1, 'GrLivArea', 4000, 12.5)
_input1 = remove_grph_outlier(_input1, 'OpenPorchSF', 500, 11.5)
_input1 = remove_grph_outlier(_input1, 'EnclosedPorch', 500)
for col in skwdlist:
    print(col)
    if _input1[col].skew() < 1:
        skwdlist.remove(col)
print(skwdlist)
for col in skwdlist:
    if _input1[col].min() > 0:
        (_input1[col], lam) = stats.boxcox(_input1[col])
        print(col, ' skew after boxcox:', _input1[col].skew())
    else:
        (_input1[col], lam) = stats.yeojohnson(_input1[col])
        print(col, ' skew after yeojohnson:', _input1[col].skew())
excllist = df_stats.index[(abs(df_stats['Corr SalePrice']) < 0.05) & (abs(df_stats['Skew']) > 1)].tolist()
print(excllist)

def bar_chart(df, feature, label):
    groups = _input1[feature].unique()
    df_grouped = _input1.groupby(feature)
    group_labels = []
    for g in groups:
        g_list = df_grouped.get_group(g)
        group_labels.append(g_list[label])
    oneway = stats.f_oneway(*group_labels)
    unique_groups = _input1[feature].unique()
    ttests = []
    for (i, group) in enumerate(unique_groups):
        for (i2, group_2) in enumerate(unique_groups):
            if i2 > i:
                type_1 = _input1[_input1[feature] == group]
                type_2 = _input1[_input1[feature] == group_2]
                if len(type_1[label]) < 2 or len(type_2[label]) < 2:
                    print("'" + group + "' n = " + str(len(type_1)) + "; '" + group_2 + "' n = " + str(len(type_2)) + '; no t-test performed')
                else:
                    (t, p) = stats.ttest_ind(type_1[label], type_2[label])
                    ttests.append([group, group_2, t.round(4), p.round(4)])
    if len(ttests) > 0:
        p_threshold = 0.05 / len(ttests)
    else:
        p_threshold = 0.05
    textstr = '            ANOVA' + '\n'
    textstr += 'F:                ' + str(oneway[0].round(2)) + '\n'
    textstr += 'p-value:          ' + str(oneway[1].round(2)) + '\n\n'
    textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'
    for ttest in ttests:
        if ttest[3] <= p_threshold:
            textstr += ttest[0] + '-' + ttest[1] + ': t=' + str(ttest[2]) + ', p=' + str(ttest[3]) + '\n'
    ax = sns.barplot(x=_input1[feature], y=_input1[label])
    ax.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

def heteroscedasticity(df, feature, label):
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.diagnostic import het_white
    import statsmodels.api as sm
    from statsmodels.formula.api import ols