import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.columns = df.columns.str.upper()

def check_df(dataframe, head=5):
    print('-' * 25, 'Shape', '-' * 25)
    print(dataframe.shape)
    print('-' * 25, 'Types', '-' * 25)
    print(dataframe.dtypes)
    print('-' * 25, 'Head', '-' * 25)
    print(dataframe.head(head))
    print('-' * 25, 'Tail', '-' * 25)
    print(dataframe.tail(head))
    print('-' * 25, 'NA', '-' * 25)
    print(dataframe.isnull().sum())
    print('-' * 25, 'Quantiles', '-' * 25)
    print(dataframe.describe([0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).T)
check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
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

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#' * 30)
    if plot:
        pass
        pass
for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.01, 0.05, 0.2, 0.5, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print('#' * 50)
    if plot:
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
for col in num_cols:
    num_summary(df, col, plot=True)

def target_variable(dataframe, target_variable, num_col):
    print(dataframe.groupby(target_variable).agg({num_col: ['mean', 'count']}), end='\n\n\n')
for col in num_cols:
    target_variable(df, 'OUTCOME', col)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return (low_limit, up_limit)

def check_outlier(dataframe, col_name):
    (low, up) = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None)
for col in num_cols:
    print(col, check_outlier(df, col))

def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr(method='spearman')
    cor_matrix = dataframe.corr(method='spearman').abs()
    uppper_triangular_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_col = [col for col in uppper_triangular_matrix.columns if any(uppper_triangular_matrix[col] > corr_th)]
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        pass
        pass
    return drop_col
drop_col = high_correlated_cols(df, plot=True, corr_th=0.8)
na_cols = ['GLUCOSE', 'BLOODPRESSURE', 'BMI', 'SKINTHICKNESS', 'INSULIN']
for col in na_cols:
    df[col].replace({0: np.nan}, inplace=True)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
msno.matrix(df, figsize=(8, 4))
msno.heatmap(df, figsize=(8, 4))
dff = pd.get_dummies(df[num_cols + cat_cols], drop_first=True)
scaler = RobustScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
for col in na_cols:
    df[col] = dff.loc[:, col]
for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
replace_with_thresholds(df, 'SKINTHICKNESS')
replace_with_thresholds(df, 'INSULIN')

def model_validation(dataframe, X, y):
    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    cv_results = cross_validate(log_model, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    scores = ['test_' + col for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    for score in scores:
        print(f'{score}: {round(cv_results[score].mean(), 4)}')
y = df['OUTCOME']
X = df.drop(['OUTCOME'], axis=1)
model_validation(df, X, y)

def bmi_converter(x):
    if x < 18.5:
        return 'Underweight'
    elif x < 25:
        return 'Normal'
    elif x < 30:
        return 'Overweight'
    else:
        return 'Obese'
df['BMI_CAT'] = df['BMI'].apply(bmi_converter)

def glucose_converter(x):
    if x < 140:
        return 'Normal'
    elif x < 200:
        return 'At_Risk'
    else:
        return 'Diabetes'
df['GLUCOSE_CAT'] = df['GLUCOSE'].apply(glucose_converter)

def bloodpressure_converter(x):
    if x < 80:
        return 'Normal'
    elif x < 90:
        return 'At_Risk'
    else:
        return 'High'
df['BLOODPRESSURE_CAT'] = df['BLOODPRESSURE'].apply(bloodpressure_converter)

def skinthickness_converter(x):
    if x <= 23:
        return 'Normal'
    else:
        return 'Not_Normal'
df['SKINTHICKNESS_CAT'] = df['SKINTHICKNESS'].apply(skinthickness_converter)
df['GLU*INS'] = df['GLUCOSE'] * df['INSULIN']
df['INS/SKIN'] = df['INSULIN'] * df['SKINTHICKNESS']
df['BMI/AGE'] = df['BMI'] / df['AGE']
df['PREGNANCIES/AGE'] = df['PREGNANCIES'] / df['AGE']
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
cat_cols = [col for col in cat_cols if col != 'OUTCOME']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X_scaled = RobustScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
X = df.drop('OUTCOME', axis=1)
y = df['OUTCOME']
model_validation(df, X, y)