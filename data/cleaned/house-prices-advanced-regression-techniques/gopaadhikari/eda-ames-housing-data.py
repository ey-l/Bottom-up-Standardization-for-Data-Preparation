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
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
label = 'SalePrice'
df.drop(columns=['Id'], inplace=True)
print('Data shape:', df.shape)
print('------------------------\nData types:\n', df.dtypes.value_counts())
print('------------------------')
df.head(10)
for col in df:
    if col[0].isdigit():
        df.rename(columns={col: 'n' + col}, inplace=True)
df['AgeBuilt'] = df['YrSold'] - df['YearBuilt']
df['AgeRemod'] = df['YrSold'] - df['YearRemodAdd']
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
df.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)

def map_ordinals(df):
    LotShape = {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}
    df.LotShape = df.LotShape.map(LotShape)
    LandContour = {'Low': 0, 'Lvl': 1, 'Bnk': 2, 'HLS': 3}
    df.LandContour = df.LandContour.map(LandContour)
    LandSlope = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
    df.LandSlope = df.LandSlope.map(LandSlope)
    ExterQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df.ExterQual = df.ExterQual.map(ExterQual)
    ExterCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df.ExterCond = df.ExterCond.map(ExterCond)
    BsmtQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df.BsmtQual = df.BsmtQual.map(BsmtQual)
    BsmtCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df.BsmtCond = df.BsmtCond.map(BsmtCond)
    BsmtExposure = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    df.BsmtExposure = df.BsmtExposure.map(BsmtExposure)
    BsmtFinType1 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    df.BsmtFinType1 = df.BsmtFinType1.map(BsmtFinType1)
    BsmtFinType2 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    df.BsmtFinType2 = df.BsmtFinType2.map(BsmtFinType2)
    HeatingQC = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df.HeatingQC = df.HeatingQC.map(HeatingQC)
    KitchenQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df.KitchenQual = df.KitchenQual.map(KitchenQual)
    Functional = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}
    df.Functional = df.Functional.map(Functional)
    FireplaceQu = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df.FireplaceQu = df.FireplaceQu.map(FireplaceQu)
    GarageFinish = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
    df.GarageFinish = df.GarageFinish.map(GarageFinish)
    GarageQual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df.GarageQual = df.GarageQual.map(GarageQual)
    GarageCond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    df.GarageCond = df.GarageCond.map(GarageCond)
    PavedDrive = {'Y': 2, 'P': 1, 'N': 0}
    df.PavedDrive = df.PavedDrive.map(PavedDrive)
    PoolQC = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0}
    df.PoolQC = df.PoolQC.map(PoolQC)
    Fence = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
    df.Fence = df.Fence.map(Fence)
    return df
df = map_ordinals(df)
print('Data shape:', df.shape)
print('------------------------\nData types:\n', df.dtypes.value_counts())
print('------------------------')
n_obs = df.shape[0]
missing_df = pd.DataFrame(columns=['Dtype', 'Missing', 'Missing ratio'])
for col in df:
    n_missing = df[col].isnull().sum()
    if n_missing:
        missing_df.loc[col] = [df[col].dtype, n_missing, n_missing / n_obs * 100]
missing_df.sort_values(by=['Dtype', 'Missing'], ascending=True)
imputer0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
df[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'GarageAge', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'PoolQC']] = imputer0.fit_transform(df[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'GarageAge', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'PoolQC']].values)
imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
df[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'KitchenQual', 'Functional']] = imputer1.fit_transform(df[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'KitchenQual', 'Functional']].values)
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'SaleType']] = imputer2.fit_transform(df[['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'SaleType']].values)
df[['GarageType']] = df[['GarageType']].replace(np.nan, 'No garage')
df[['Alley']] = df[['Alley']].replace(np.nan, 'No alley')
df[['MiscFeature']] = df[['MiscFeature']].replace(np.nan, 'None')
df.dropna(axis=1, inplace=True)
print('Data shape after missing value imputation: ', df.shape)
df

def num_univarstats_r(df, numeric=True):
    corr_col = 'Corr ' + label
    univar_df = pd.DataFrame(columns=['Dtype', 'Count', 'Unique', 'Mode', 'Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Std', 'Skew', 'Kurt', corr_col])
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            univar_df.loc[col] = [df[col].dtype, df[col].count(), df[col].nunique(), df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75), df[col].max(), df[col].std(), df[col].skew(), df[col].kurt(), df[col].corr(df[label])]
    return univar_df
df_stats = num_univarstats_r(df).sort_values(by=['Dtype', 'Corr SalePrice', 'Skew'], ascending=False)
df_stats
df[label] = np.log(df[label])
print('SalePrice Skewness:', df[label].skew(), ' Kurtosis:', df[label].kurt())
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121)
sns.histplot(data=df, x=label, kde=True, ax=ax)
ax = fig.add_subplot(122)
stats.probplot(df[label], plot=ax)
skwdlist = df_stats.index[(round(abs(df_stats['Corr SalePrice']), 1) > 0.1) & (abs(df_stats['Corr SalePrice']) != 1) & (round(abs(df_stats['Skew'])) > 1)].tolist()
print(skwdlist)

def remove_num_outlier(df, col):
    Q1 = np.percentile(df[col], 25, interpolation='midpoint')
    Q3 = np.percentile(df[col], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    print('Old Shape: ', df.shape)
    upper = np.where(df[col] >= Q3 + 1.5 * IQR)
    lower = np.where(df[col] <= Q1 - 1.5 * IQR)
    df.drop(upper[0], inplace=True)
    df.drop(lower[0], inplace=True)
    print('New Shape: ', df.shape)

def remove_grph_outlier(df, col, xlim, ylim=None):
    print(col, ' Skew before:', df[col].skew())
    if ylim:
        df = df[~((df[col] > xlim) & (df[label] < ylim))]
    else:
        df = df[df[col] < xlim]
    print(col, ' Skew after:', df[col].skew())
    print(df.shape)
    return df
df = remove_grph_outlier(df, 'LotFrontage', 300)
df = remove_grph_outlier(df, 'LotArea', 100000)
df = remove_grph_outlier(df, 'GrLivArea', 4000, 12.5)
df = remove_grph_outlier(df, 'OpenPorchSF', 500, 11.5)
df = remove_grph_outlier(df, 'EnclosedPorch', 500)
for col in skwdlist:
    print(col)
    if df[col].skew() < 1:
        skwdlist.remove(col)
print(skwdlist)
for col in skwdlist:
    if df[col].min() > 0:
        (df[col], lam) = stats.boxcox(df[col])
        print(col, ' skew after boxcox:', df[col].skew())
    else:
        (df[col], lam) = stats.yeojohnson(df[col])
        print(col, ' skew after yeojohnson:', df[col].skew())
excllist = df_stats.index[(abs(df_stats['Corr SalePrice']) < 0.05) & (abs(df_stats['Skew']) > 1)].tolist()
print(excllist)

def bar_chart(df, feature, label):
    groups = df[feature].unique()
    df_grouped = df.groupby(feature)
    group_labels = []
    for g in groups:
        g_list = df_grouped.get_group(g)
        group_labels.append(g_list[label])
    oneway = stats.f_oneway(*group_labels)
    unique_groups = df[feature].unique()
    ttests = []
    for (i, group) in enumerate(unique_groups):
        for (i2, group_2) in enumerate(unique_groups):
            if i2 > i:
                type_1 = df[df[feature] == group]
                type_2 = df[df[feature] == group_2]
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
    ax = sns.barplot(x=df[feature], y=df[label])
    ax.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)


def heteroscedasticity(df, feature, label):
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.diagnostic import het_white
    import statsmodels.api as sm
    from statsmodels.formula.api import ols