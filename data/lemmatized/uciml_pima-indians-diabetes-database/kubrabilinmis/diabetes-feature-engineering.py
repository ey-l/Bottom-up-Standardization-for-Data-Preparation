import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df_ = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = df_.copy()

def check_df(dataframe):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(3))
    print('##################### Tail #####################')
    print(dataframe.tail(3))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        pass
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        pass

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        pass
        pass

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(), 'Count': temp_df.groupby(col)[target].count()}), end='\n\n\n')
df.columns = [col.upper() for col in df.columns]
check_df(df)
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
cat_summary(df, 'OUTCOME', plot=True)
for col in num_cols:
    num_summary(df, col, plot=True)
target_summary_with_num(df, 'OUTCOME', col)
pass
pass
ax.set_title('Correlation Matrix', fontsize=20)
df.isnull().sum()
df[['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']] = df[['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']].replace(0, np.NaN)
na_cols = missing_values_table(df, True)
import missingno as msno
msno.bar(df)
missing_vs_target(df, 'OUTCOME', na_cols)

def median_target(variable):
    temp = df[df[variable].notnull()]
    temp = temp[[variable, 'OUTCOME']].groupby(['OUTCOME'])[[variable]].median().reset_index()
    return temp
columns = df.columns
columns = columns.drop('OUTCOME')
for col in columns:
    df.loc[(df['OUTCOME'] == 0) & df[col].isnull(), col] = median_target(col)[col][0]
    df.loc[(df['OUTCOME'] == 1) & df[col].isnull(), col] = median_target(col)[col][1]
for col in num_cols:
    print(col, check_outlier(df, col))
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
th = np.sort(df_scores)[5]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T
df[df_scores < th].index
df.drop(axis=0, labels=df[df_scores < th].index)
df = df.drop(axis=0, labels=df[df_scores < th].index)
df.head()
df.shape
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
df['NEW_GLUCOSE'] = pd.cut(x=df['GLUCOSE'], bins=[0, 140, 200, 300], labels=['Normal', 'Prediabetes', 'Diabetes'])
df['AGE'].min()
df['AGE'].max()
df.loc[df['AGE'] <= 30, 'NEW_AGE'] = 'young'
df.loc[(df['AGE'] > 30) & (df['AGE'] <= 50), 'NEW_AGE'] = 'middle_age'
df.loc[df['AGE'] > 50, 'NEW_AGE'] = 'old'
df.head()
df['NEW_AGE'].value_counts()
df.loc[df['BLOODPRESSURE'] < 70, 'NEW_BLOOD_CAT'] = 'hipotansiyon'
df.loc[(df['BLOODPRESSURE'] >= 70) & (df['BLOODPRESSURE'] < 90), 'NEW_BLOOD_CAT'] = 'normal'
df.loc[df['BLOODPRESSURE'] >= 90, 'NEW_BLOOD_CAT'] = 'hipertansiyon'
df['NEW_INSULIN'] = pd.cut(x=df['INSULIN'], bins=[0, 140, 200, df['INSULIN'].max()], labels=['Normal', 'Hidden_diabetes', 'Diabetes'])
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()
y = df['OUTCOME']
X = df.drop(['OUTCOME'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)
from sklearn.ensemble import RandomForestClassifier