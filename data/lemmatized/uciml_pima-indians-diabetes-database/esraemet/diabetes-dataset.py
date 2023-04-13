import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.Pregnancies.dtypes

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
     It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
     Note: Categorical variables with numerical appearance are also included in categorical variables.
     parameters
     -------
         dataframe: dataframe
             The dataframe from which variable names are to be retrieved
         cat_th: int optional
             class threshold for numeric but categorical variables
         car_th: int, optional
             class threshold for categorical but cardinal variables
     Returns
     -------
         cat_cols: list
             Categorical variable list
         num_cols: list
             Numeric variable list
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

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)
for col in num_cols:
    print(col, check_outlier(df, col))
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))
(low, up) = outlier_thresholds(df, 'Insulin')
df[(df['Insulin'] < low) | (df['Insulin'] > up)].shape
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T
df[df_scores < th].index
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
target_summary_with_cat(df, 'Insulin', 'Age')
for col in cat_cols:
    target_summary_with_cat(df, 'Insulin', col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')
target_summary_with_num(df, 'Insulin', 'Age')
for col in num_cols:
    target_summary_with_num(df, 'Insulin', col)
num_cols
for col in num_cols:
    print(col, check_outlier(df, col))
pass

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for i in df.columns:
    (low, up) = outlier_thresholds(df, i)
    if ((df[i] < low) | (df[i] > up)).any():
        print(f'\nIndices: {df[(df[i] < low) | (df[i] > up)].index}\n')
        print(df[(df[i] < low) | (df[i] > up)].head())
        replace_with_thresholds(df, i)
outlier_thresholds(df, 'Age')
(low, up) = outlier_thresholds(df, 'Age')
df[(df['Age'] < low) | (df['Age'] > up)].head()

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, 'Age')
df.isnull().values.any()
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()
df.corr()
df.shape
df.head()
pass
pass

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
df.head()
cols = ['Pregnancies', 'Glucose', 'Insulin']
for i in cols:
    df[i] = df[i].replace({'0': np.nan, 0: np.nan})
df[cols].head()
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != 'O' else x, axis=0)
df.info()
df.head()
df.columns = [col.upper() for col in df.columns]
df['NEW_PREGNANCIES_BOOL'] = df['PREGNANCIES'].notnull().astype('int')
df.groupby('NEW_PREGNANCIES_BOOL').agg({'AGE': 'mean'})
df.loc[df['GLUCOSE'] + df['INSULIN'] > 0, 'DIET'] = 1
df.loc[df['GLUCOSE'] + df['INSULIN'] <= 0, 'DIET'] = 0
df.groupby('DIET').agg({'AGE': 'mean'})
df['NEW_AGE_INSULIN'] = df['AGE'] * df['INSULIN']
df['INSULIN_OUTCOME'] = df['INSULIN'] * df['OUTCOME']
df.head()
df['INSULIN_OUTCOME'] = df['INSULIN'].notnull().astype('int')
df.groupby('INSULIN_OUTCOME').agg({'OUTCOME': 'mean'})
df['INSULIN_PREGNANCIES'] = df['INSULIN'].notnull().astype('int')
df.groupby('INSULIN_PREGNANCIES').agg({'PREGNANCIES': 'mean'})
df.DIET.value_counts()
df.head()
df['DIET'].head()
le = LabelEncoder()
le.fit_transform(df['DIET'])[0:5]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df.head()
y = df['OUTCOME']
X = df.drop(['OUTCOME'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)