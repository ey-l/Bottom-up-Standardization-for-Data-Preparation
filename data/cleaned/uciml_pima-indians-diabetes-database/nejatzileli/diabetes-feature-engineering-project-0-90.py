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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = data.copy()

def check_df(data, x=5):
    print('################################# shape ##########################')
    print(data.shape)
    print('################################# type ##########################')
    print(data.dtypes)
    print('################################# head ##########################')
    print(data.head(x))
    print('################################# tail ##########################')
    print(data.tail(x))
    print('################################# null ##########################')
    print(data.isnull().sum().sort_values(ascending=False))
    print('################################# quantiles #####################')
    print(data.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)

def grab_col_names(dataframe, categorical=10, cardinal=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype == 'O' and dataframe[col].nunique() > cardinal]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype != 'O' and dataframe[col].nunique() < categorical]
    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != 'O' and col not in num_but_cat]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return (cat_cols, num_cols, cat_but_car)
(cat_cols, num_cols, cat_but_car) = grab_col_names(df, categorical=10, cardinal=20)

def num_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numeric_col].describe(quantiles).T)
    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)

for col in num_cols:
    num_summary(df, col, plot=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#####################', col_name, '############################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)

for col in cat_cols:
    cat_summary(df, col, plot=True)
df.groupby('Pregnancies').agg({'Outcome': ['mean', 'sum']}).reset_index().sort_values(by=('Outcome', 'sum'), ascending=False)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}))
for col in num_cols:
    target_summary_with_num(df, 'Outcome', col)

def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
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

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    (low, up) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
    else:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
        return outlier_index
num_cols
grab_outliers(df, 'Pregnancies', True)
grab_outliers(df, 'Glucose', True)
grab_outliers(df, 'BloodPressure', True)
grab_outliers(df, 'Insulin', True)
grab_outliers(df, 'BMI', True)
grab_outliers(df, 'DiabetesPedigreeFunction', True)
grab_outliers(df, 'Age', True)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap='RdBu')

df.corr()
df.isnull().sum()
df.describe().T
df[df['BloodPressure'] < 60]['BloodPressure'].count()
df[df['BloodPressure'] < 60].index
df[df['BMI'] < 12]['BMI'].count()
df[df['BMI'] < 12]['BMI'].index
df[df['Insulin'] == 0]['Insulin'].count()
df[df['Insulin'] == 0]['Insulin'].index
df[df['SkinThickness'] == 0]['SkinThickness'].count()
df[df['SkinThickness'] == 0]['SkinThickness'].index
df[(df['BloodPressure'] < 60) & (df['Insulin'] == 0) & (df['SkinThickness'] == 0)].count()
df['BloodPressure'].replace(0, np.nan, inplace=True)
df['BMI'].replace(0, np.nan, inplace=True)
df['Insulin'].replace(0, np.nan, inplace=True)
df['SkinThickness'].replace(0, np.nan, inplace=True)
df['Glucose'].replace(0, np.nan, inplace=True)
df.isnull().sum()
missing_values_table(df)
na_cols = [col for col in df.columns if df[col].isnull().sum() != 0]

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(), 'Count': temp_df.groupby(col)[target].count()}), end='\n\n\n')
missing_vs_target(df, 'Outcome', na_cols)
df.groupby('Outcome').agg({'SkinThickness': 'median'})
df['SkinThickness'].fillna(df.groupby('Outcome')['SkinThickness'].transform('median'), inplace=True)
df.isnull().sum()
df.groupby('Outcome').agg({'Insulin': 'median'})
df['Insulin'].fillna(df.groupby('Outcome')['Insulin'].transform('median'), inplace=True)
df.isnull().sum()
df['BloodPressure'].fillna(df.groupby('Outcome')['BloodPressure'].transform('median'), inplace=True)
df.isnull().sum()
df['BMI'].fillna(df.groupby('Outcome')['BMI'].transform('median'), inplace=True)
df.isnull().sum()
df['Glucose'].fillna(df.groupby('Outcome')['Glucose'].transform('median'), inplace=True)
df.isnull().sum()
df['GLUCOSE_CAT_NEW'] = pd.cut(x=df['Glucose'], bins=[-1, 80, 140, 160, 200], labels=['Hypoglecimia', 'Normal', 'Impaired_Glucose', 'Diabetic_Glucose'])
df['AGE_CATEGORIES_NEW'] = pd.cut(x=df['Age'], bins=[18, 44, 64, 100], labels=['Adults', 'Matures', 'Boomers'])
df['DIASTOLIC_BLOOD_PRESSURE_NEW'] = pd.cut(x=df['BloodPressure'], bins=[0, 80, 89, 120, 300], labels=['Normal', 'Norm_Check_Sylostic', 'Hypertension', 'Hypertension_Crisis'])
df['INSULIN_NEW'] = pd.cut(x=df['Insulin'], bins=[0, 120, 1000], labels=['Normal', 'Abnormal'])
df['BMI_CAT_NEW'] = pd.cut(x=df['BMI'], bins=[0, 18, 25, 29, 68], labels=['Underweight', 'Normal', 'Overweight', 'Obesity'])
df['Pregnancies'].describe()
df.loc[df['Pregnancies'] == 0, 'PREGNANT_CAT_NEW'] = 'NO_TIME'
df.loc[df['Pregnancies'] == 1, 'PREGNANT_CAT_NEW'] = 'ONE_TIME'
df.loc[df['Pregnancies'] > 1, 'PREGNANT_CAT_NEW'] = 'MANY_TIME'
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
df['NEW_GLUCOSE*INSULIN'] = df['Glucose'] * df['Insulin']
df['Age/Numberofpregnancy'] = df['Pregnancies'] * df['Age']
target_summary_with_num(df, 'Outcome', 'Age/Numberofpregnancy')
df['Age/Numberofpregnancy'].describe()
df['Age/Numberofpregnancy_CAT'] = pd.cut(x=df['Age/Numberofpregnancy'], bins=[0, 125, 170, 800])
df.groupby('Age/Numberofpregnancy_CAT').agg({'Outcome': ['mean', 'sum']}).reset_index().sort_values(by=('Outcome', 'sum'), ascending=False)
df.head()
(cat_cols, num_cols, cat_but_car) = grab_col_names(df, categorical=10, cardinal=20)
num_cols
grab_outliers(df, 'Glucose', True)
grab_outliers(df, 'BloodPressure', True)
grab_outliers(df, 'Insulin', True)
grab_outliers(df, 'BMI', True)
grab_outliers(df, 'DiabetesPedigreeFunction', True)
grab_outliers(df, 'Age', True)
grab_outliers(df, 'Age/Numberofpregnancy', True)
x = 18 / df.shape[0]
df.shape
replace_with_thresholds(df, 'Insulin')
replace_with_thresholds(df, 'DiabetesPedigreeFunction')
df.shape
df = df.copy()
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]
binary_cols
for col in binary_cols:
    df = label_encoder(df, col)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ['Outcome']]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)
from sklearn.ensemble import RandomForestClassifier