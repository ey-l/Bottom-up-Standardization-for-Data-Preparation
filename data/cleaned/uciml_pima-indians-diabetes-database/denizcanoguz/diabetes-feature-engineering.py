import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
from _plotly_utils import png
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
df_ = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = df_.copy()
df.columns = [col.upper() for col in df.columns]

def check_df(dataframe, head=5, tail=5, quan=False):
    print('##################### Shape #####################')
    print(dataframe.shape, '\n')
    print('##################### Types #####################')
    print(dataframe.dtypes, '\n')
    print('##################### Head #####################')
    print(dataframe.head(head), '\n')
    print('##################### Tail #####################')
    print(dataframe.tail(tail), '\n')
    print('##################### NA #####################')
    print(dataframe.isnull().sum(), '\n')
    if quan:
        print('##################### Quantiles #####################')
        print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)

def grab_col_names(dataframe, cat_th=3, car_th=10):
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
num_cols

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print('\n')
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)

        print('\n')
for col in num_cols:
    num_summary(df, col, plot=True)
cat_cols

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)

cat_summary(df, 'OUTCOME')
df.groupby('OUTCOME').agg({'PREGNANCIES': 'mean', 'GLUCOSE': 'mean', 'BLOODPRESSURE': 'mean', 'SKINTHICKNESS': 'mean', 'INSULIN': 'mean', 'BMI': 'mean', 'DIABETESPEDIGREEFUNCTION': 'mean', 'AGE': 'mean'}).sort_values('OUTCOME')
sns.set_theme(style='whitegrid')
sns.boxplot(data=df, orient='h', palette='Set2')


def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)
for col in num_cols:
    print(col, outlier_thresholds(df, col))

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
cor = df.corr(method='pearson')
cor
sns.heatmap(cor, annot=True)

df['GLUCOSE'].replace({0: np.nan}, inplace=True)
df['BLOODPRESSURE'].replace({0: np.nan}, inplace=True)
df['SKINTHICKNESS'].replace({0: np.nan}, inplace=True)
df['INSULIN'].replace({0: np.nan}, inplace=True)
df['BMI'].replace({0: np.nan}, inplace=True)
df['AGE'].replace({0: np.nan}, inplace=True)
msno.heatmap(df)

missing_values_table(df)
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()
scaler = RobustScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=33)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()
df['INSULIN'] = dff[['INSULIN']]
df['SKINTHICKNESS'] = dff[['SKINTHICKNESS']]
df['BLOODPRESSURE'] = dff[['BLOODPRESSURE']]
df['BMI'] = dff[['BMI']]
df['GLUCOSE'] = dff[['GLUCOSE']]
missing_values_table(df)
clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:30]
np.sort(df_scores)[0:30]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')

th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T
clf_index = df[df_scores < th].index
df.drop(index=clf_index, inplace=True)
df['INSULIN/AGE'] = df['INSULIN'] / df['AGE']
df['BMI/AGE'] = df['BMI'] / df['AGE']
df['PREGNANCIES/AGE'] = df['PREGNANCIES'] / df['AGE']
df['INS*GLU'] = df['INSULIN'] * df['GLUCOSE']
df.drop(['AGE'], axis=1, inplace=True)
df['NEW_BMI_CAT'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['UNDER WEIGHT', 'NORMALY WEIGHT', 'OVER WEIGHT', 'OBESE'])
df.groupby('NEW_BMI_CAT')['OUTCOME'].mean()
df['NEW_BP_CAT'] = pd.cut(x=df['BLOODPRESSURE'], bins=[0, 80, 84, 90, 122], labels=['OPTIMAL PRESSURE', 'NORMAL PRESSURE', 'HIGH-NORMAL PRESSURE', 'HIGH PRESSURE'])
df.groupby('NEW_BP_CAT')['OUTCOME'].mean()
df.head()
df.shape
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(), 'RATIO': dataframe[col].value_counts() / len(dataframe), 'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')
rare_analyser(df, 'OUTCOME', cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()
y = df['OUTCOME']
X = df.drop(['OUTCOME'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier