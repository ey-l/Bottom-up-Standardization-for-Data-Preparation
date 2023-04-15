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
import statsmodels.api as sm
import warnings
warnings.simplefilter('ignore')
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df_ = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df = df_.copy()
df.head()
df.shape
df.info()

def check_df(dataframe, head=5, tail=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head ######################')
    print(dataframe.head(head))
    print('##################### Tail ######################')
    print(dataframe.tail(tail))
    print('##################### NA ########################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df.columns = col_names
col_names
df['Glucose'].value_counts()
df.info()
for col in col_names:
    print(df[col].value_counts())
df.isnull().sum()
X = df.drop(['Age'], axis=1)
y = df['Age']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
(X_train.shape, X_test.shape)
X_train.dtypes
X_train.head()
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Outcome'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()
X_test.head()
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def check_df(dataframe, head=5, tail=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head ######################')
    print(dataframe.head(head))
    print('##################### Tail ######################')
    print(dataframe.tail(tail))
    print('##################### NA ########################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
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
cat_cols
num_cols
df[num_cols].describe().T
df[cat_cols].describe()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)
outlier_thresholds(df, num_cols)
outlier_thresholds(df, cat_cols)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, num_cols)
check_outlier(df, cat_cols)
for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    (low, up) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
    else:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
        return outlier_index
grab_outliers(df, num_cols)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')
for col in df.columns:
    target_summary_with_num(df, 'Outcome', col)
cor = df.corr(method='pearson')
cor
sns.heatmap(cor)

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')

th = np.sort(df_scores)[3]
th

def assing_missing_values(dataframe, except_cols):
    for col in dataframe.columns:
        dataframe[col] = [val if val != 0 or col in except_cols else np.nan for val in df[col].values]
    return dataframe
df = assing_missing_values(df, except_cols=['Pregnancies', 'Outcome'])
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
na_cols = missing_values_table(df, True)
msno.bar(df)

msno.heatmap(df)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_Na_Flag'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_Na_')].columns
    for col in na_flags:
        print(pd.DataFrame({'Target_Mean': temp_df.groupby(col)[target].mean(), 'Count': temp_df.groupby(col)[target].count()}), end='\n\n\n')
missing_vs_target(df, 'Outcome', na_cols)
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
df.head()
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()
imputer = KNNImputer()
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)
df.loc[df['Pregnancies'] >= 10, 'New_Preg_Cat'] = 0.0
df.loc[df['Pregnancies'] == 0, 'New_Preg_Cat'] = 'Not Pregnant'
df.loc[df['Pregnancies'] > 0, 'New_Preg_Cat'] = 'Pregnant'
df.loc[df['Glucose'] < 70, 'New_Glucose_Cat'] = 'low'
df.loc[(df['Glucose'] < 100) & (df['Glucose'] >= 70), 'New_Glucose_Cat'] = 'Normal'
df.loc[(df['Glucose'] < 125) & (df['Glucose'] >= 100), 'New_Glucose_Cat'] = 'Potential'
df.loc[df['Glucose'] >= 125, 'New_Glucose_Cat'] = 'High'
df.loc[df['BloodPressure'] > 90, 'New_Bloodpr_Cat'] = 'High'
df.loc[(df['BloodPressure'] <= 90) & (df['BloodPressure'] > 0), 'New_Bloodpr_Cat'] = 'Normal'
df.loc[df['BMI'] < 18.5, 'New_BMI_Cat'] = 'Underweight'
df.loc[(df['BMI'] < 30) & (df['BMI'] >= 18.5), 'New_BMI_Cat'] = 'Normal'
df.loc[(df['BMI'] < 34.9) & (df['BMI'] >= 30), 'New_BMI_Cat'] = 'Obese'
df.loc[df['BMI'] >= 34.9, 'New_BMI_Cat'] = 'Extremely Obese'
df.loc[df['Age'] <= 21, 'New_Age_Cat'] = 'Young'
df.loc[(df['Age'] <= 50) & (df['Age'] > 21), 'New_Age_Cat'] = 'Mature'
df.loc[df['Age'] > 50, 'New_Age_Cat'] = 'Senior'
df.head()
df.isnull().sum()
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

def one_hot_encoder(dataframe, categorical_columns, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(), 'RATIO': dataframe[col].value_counts() / len(dataframe), 'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]
    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])
    return dataframe
(cat_cols, num_cols, cat_but_car) = grab_col_names(df)
rare_analyser(df, 'Outcome', cat_cols)
df.head()
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols
df = one_hot_encoder(df, ohe_cols)
df.head()
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
useless_cols
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.head()
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=47)