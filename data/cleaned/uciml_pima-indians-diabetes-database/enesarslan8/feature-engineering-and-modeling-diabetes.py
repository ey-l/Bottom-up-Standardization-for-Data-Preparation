import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.columns = df.columns.str.lower()
df.columns

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the categorical, numerical and categorical but cardinal variables.
    Note: Categorical variables includes numerical values that have low unique values than 10.

    Parameters
    ------
        dataframe: dataframe
                Dataframe that wanted to get column types
        cat_th: int, optional
                Threshold value for the numerical but categorical values
        car_th: int, optinal
                Threshold value for the categorical but cardinal values

    Returns
    ------
        cat_cols: list
                Categorical variables
        num_cols: list
                Numeric variables
        cat_but_car: list
                Categorical but cardinal values

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return (cat_cols, num_cols, cat_but_car)
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
print('Categorical: {}\nNumerical: {}\nCardinal: {}'.format(cat_cols, num_cols, cat_but_car))
plt.rcParams['figure.figsize'] = (20, 20)
for (index, num_col) in enumerate(num_cols):
    num_group_df = df.groupby('outcome').agg({num_col: 'mean'}).reset_index()
    plt.subplot(4, 4, index + 1)
    sns.barplot(x='outcome', y=num_col, data=num_group_df)
plt.rcParams['figure.figsize'] = (20, 20)
for (index, num_col) in enumerate(num_cols):
    plt.subplot(4, 4, index + 1)
    sns.boxplot(x=num_col, data=df)

sns.pairplot(hue='outcome', data=df)


def desc_stats(dataframe):
    desc = dataframe.describe().T
    desc_df = pd.DataFrame(index=[col for col in dataframe.columns], columns=desc.columns, data=desc)
    (f, ax) = plt.subplots(figsize=(10, desc_df.shape[0] * 0.78))
    sns.heatmap(desc_df, annot=True, cmap='Reds', fmt='.2f', ax=ax, linewidths=2.6, cbar=False, annot_kws={'size': 14})
    plt.xticks(size=18)
    plt.yticks(size=14, rotation=0)
    plt.title('Descriptive Statistics', size=16)

desc_stats(df[num_cols])
df.info()
df.isnull().sum()
min_zero_cols = df.columns[df.min() == 0]
cant_be_zero_cols = min_zero_cols.drop(['outcome', 'pregnancies'])
cant_be_zero_cols
df[cant_be_zero_cols] = df[cant_be_zero_cols].applymap(lambda x: np.nan if x == 0 else x)
df.head()
msno.matrix(df)

df.corrwith(df.outcome)
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
df = df.apply(lambda x: x.fillna(df.groupby('outcome')[x.name].transform('median')))
missing_values_table(df)
df.describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T

def outlier_thresholds(dataframe, colname, q1=0.05, q3=0.95):
    quartile1 = dataframe[colname].quantile(q1)
    quartile3 = dataframe[colname].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)

def check_outlier(low_limit, up_limit, dataframe, colname):
    if dataframe[(dataframe[colname] > up_limit) | (dataframe[colname] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(low_limit, up_limit, dataframe, colname):
    dataframe.loc[dataframe[colname] > up_limit, colname] = up_limit
    dataframe.loc[dataframe[colname] < low_limit, colname] = low_limit
outlier_cols = []
for col in num_cols:
    (low_limit, up_limit) = outlier_thresholds(df, col)
    if check_outlier(low_limit, up_limit, df, col):
        outlier_cols.append(col)
outlier_cols
for col in outlier_cols:
    (low, up) = outlier_thresholds(df, col)
    print(col, up)
    replace_with_thresholds(low, up, df, col)
(df.bloodpressure.mean(), df.bloodpressure.median())
df['IS_BP_NORMAL'] = df.bloodpressure <= 80
df['IS_BP_NORMAL']
df.groupby('IS_BP_NORMAL').agg({'outcome': ['mean', 'count']})
from statsmodels.stats.proportion import proportions_ztest
(_, pvalue) = proportions_ztest(count=[df.loc[df.IS_BP_NORMAL == True, 'outcome'].sum(), df.loc[df.IS_BP_NORMAL == False, 'outcome'].sum()], nobs=[df.loc[df.IS_BP_NORMAL == True, 'outcome'].shape[0], df.loc[df.IS_BP_NORMAL == False, 'outcome'].shape[0]])
pvalue < 0.005
df['NO_KID'] = df.pregnancies == 0
df['NO_KID']
df.groupby('NO_KID').agg({'outcome': 'mean'})
'19-24 Age: 19-24 BMI.\n25-34 Age: 20-25 BMI.\n35-44 Age: 21-26 BMI.\n45-54 Age: 22-27 BMI.\n55-64 Age: 23-28 BMI.\n65 Age or above: 24-29 BMI.'

def is_bmi_normal(age, bmi):
    if 19 <= age <= 24 and 19 <= bmi <= 24:
        return True
    elif 25 <= age <= 34 and 20 <= bmi <= 25:
        return True
    elif 35 <= age <= 44 and 21 <= bmi <= 26:
        return True
    elif 45 <= age <= 54 and 22 <= bmi <= 27:
        return True
    elif 55 <= age <= 64 and 23 <= bmi <= 28:
        return True
    elif 65 <= age and 24 <= bmi <= 29:
        return True
    else:
        return False
df['IS_BMI_NORMAL'] = df.apply(lambda x: is_bmi_normal(x.age, x.bmi), axis=1)
df.groupby('IS_BMI_NORMAL').agg({'outcome': ['count', 'mean']})
from statsmodels.stats.proportion import proportions_ztest
(_, pvalue) = proportions_ztest(count=[df.loc[df.IS_BMI_NORMAL == True, 'outcome'].sum(), df.loc[df.IS_BMI_NORMAL == False, 'outcome'].sum()], nobs=[df.loc[df.IS_BMI_NORMAL == True, 'outcome'].shape[0], df.loc[df.IS_BMI_NORMAL == False, 'outcome'].shape[0]])
pvalue < 0.005
df['PREG_INTERVAL'] = pd.qcut(df.pregnancies, 5)
df.groupby('PREG_INTERVAL').agg({'outcome': ['count', 'mean']})
df['AGE_INTERVAL'] = pd.qcut(df.age, 5)
df.groupby('AGE_INTERVAL').agg({'outcome': ['count', 'mean']})
bmi_labels = ['Underweight', 'Healthy Weight', 'Overweight', 'Obesity']
df['BMI_Cat'] = pd.cut(df['bmi'], [0, 18.5, 25, 30, df['bmi'].max()], labels=bmi_labels)
df.dtypes
dff = pd.get_dummies(df, drop_first=True)
dff.head()
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
Y = dff.outcome
X = dff.drop('outcome', axis=1)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=10)
(X_train.shape, Y_train.shape)