import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()

def check_df(dataframe, head=5):
    print('######################### Head #########################')
    print(dataframe.head(head))
    print('######################### Tail #########################')
    print(dataframe.tail(head))
    print('######################### Shape #########################')
    print(dataframe.shape)
    print('######################### Types #########################')
    print(dataframe.dtypes)
    print('######################### NA #########################')
    print(dataframe.isnull().sum())
    print('######################### Qurtiles #########################')
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['uint8', 'int64', 'float64']]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ['category', 'object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['uint8', 'int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]
    return (cat_cols, num_cols, cat_but_car)
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
print(f'Observations: {df.shape[0]}')
print(f'Variables: {df.shape[1]}')
print(f'Cat_cols: {len(cat_cols)}')
print(f'Num_cols: {len(num_cols)}')
print(f'Cat_but_car: {len(cat_but_car)}')

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ration': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)


def cat_summary_df(dataframe):
    (cat_cols, num_cols, cat_but_car) = grab_col_names(df)
    for col in cat_cols:
        cat_summary(dataframe, col, plot=True)
cat_summary_df(df)

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)
    if plot:
        dataframe[num_col].hist(bins=20)
        plt.xlabel(num_col)
        plt.title(num_col)


def num_summary_df(dataframe):
    (cat_cols, num_cols, cat_but_car) = grab_col_names(df)
    for col in num_cols:
        num_summary(dataframe, col, plot=True)
num_summary_df(df)

def plot_num_summary(dataframe):
    (cat_cols, num_cols, cat_but_car) = grab_col_names(dataframe)
    plt.figure(figsize=(12, 4))
    for (index, col) in enumerate(num_cols):
        plt.subplot(2, 4, index + 1)
        plt.tight_layout()
        dataframe[col].hist(bins=20)
        plt.title(col)
plot_num_summary(df)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}))
    print('#############################################')

def target_summary_with_num_df(dataframe, target):
    (cat_cols, num_cols, cat_but_car) = grab_col_names(df)
    for col in num_cols:
        target_summary_with_num(dataframe, target, col)
target_summary_with_num_df(df, 'Outcome')
df.corr()
(f, ax) = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax, cmap='magma')
ax.set_title('Correlation Matrix', fontsize=20)


def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if drop_list == []:
        print("############## After Correlation Analysis, You Don't Need to Remove Variables ##############")
    if plot:
        sns.set(rc={'figure.figsize': (18, 13)})
        sns.heatmap(corr, cmap='RdBu')

    return drop_list
high_correlated_cols(df, plot=True)

def exploratory_data(dataframe):
    import warnings
    warnings.filterwarnings('ignore')
    (cat_cols, num_cols, cat_but_car) = grab_col_names(dataframe)
    (fig, ax) = plt.subplots(8, 3, figsize=(30, 90))
    sns.set(font_scale=2)
    for (index, col) in enumerate(num_cols):
        sns.distplot(dataframe[col], ax=ax[index, 0])
        sns.boxplot(dataframe[col], ax=ax[index, 1])
        stats.probplot(dataframe[col], plot=ax[index, 2])
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle('Visualizing Continuous Columns')
exploratory_data(df)
df.isnull().sum()
zero_columns = [col for col in df.columns if df[col].min() == 0 and col not in ['Pregnancies', 'Outcome']]
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])
df.isnull().sum()

def missing_value_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
na_columns = missing_value_table(df, na_name=True)

def show_missing_value_plot(dataframe, bar=True, matrix=True, heatmap=True):

    import missingno as msno
    if bar:
        msno.bar(dataframe)
    if matrix:
        msno.matrix(dataframe)
    if heatmap:
        msno.heatmap(dataframe)
show_missing_value_plot(df[na_columns])

def missing_vs_target(dataframe, target):
    na_columns = missing_value_table(dataframe, na_name=True)
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(), 'Count': temp_df.groupby(col)[target].count()}))
        print('##################################################')
missing_vs_target(df, 'Outcome')
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()
df.isnull().sum()

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquartile_range
    up_limit = quartile3 + 1.5 * interquartile_range
    return (low_limit, up_limit)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)
for col in df.columns:
    print(col, check_outlier(df, col))
df.loc[(df['Age'] >= 21) & (df['Age'] < 50), 'NEW_AGE_CAT'] = 'mature'
df.loc[df['Age'] >= 50, 'NEW_AGE_CAT'] = 'senior'
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
df.loc[(df['BMI'] < 18.5) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'underweightmature'
df.loc[(df['BMI'] < 18.5) & (df['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'underweightsenior'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'healthymature'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25) & (df['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'healthysenior'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'overweightmature'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30) & (df['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'overweightsenior'
df.loc[(df['BMI'] > 18.5) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'obesemature'
df.loc[(df['BMI'] > 18.5) & (df['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'obesesenior'
df.loc[(df['Glucose'] < 70) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'lowmature'
df.loc[(df['Glucose'] < 70) & (df['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'lowsenior'
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'normalmature'
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100) & (df['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'normalsenior'
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] <= 125) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'hiddenmature'
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] <= 125) & (df['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'hiddensenior'
df.loc[(df['Glucose'] > 125) & ((df['Age'] >= 21) & (df['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'highmature'
df.loc[(df['Glucose'] > 125) & (df['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'highsenior'

def set_insulin(dataframe, col_name='Insulin'):
    if 16 <= dataframe[col_name] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'
df['NEW_INSULIN_SCORE'] = df.apply(set_insulin, axis=1)
df['NEW_GLUCOSE*INSULIN'] = df['Glucose'] * df['Insulin']
df['NEW_GLUCOSE*PREGNANCIES'] = df['Glucose'] * df['Pregnancies']
df.columns = [col.upper() for col in df.columns]
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ['OUTCOME']]

def one_hot_encoding(dataframe, cat_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoding(df, cat_cols, drop_first=True)
df.head()

def Logistic_Regression_model(dataframe, target, cv=10, results=False, conf_matrix=False, c_report=False, roc=False):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)