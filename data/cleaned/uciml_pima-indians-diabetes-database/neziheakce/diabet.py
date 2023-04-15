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

def load_diabets():
    data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
    return data
df = load_diabets()
df.describe().T

def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold value for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = the total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

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
df['Outcome'].value_counts()
sns.countplot(x='Outcome', data=df)


def target_summary(dataframe, target, cat_cols, num_cols):
    for col in dataframe:
        print(col, ':', len(dataframe[col].value_counts()))
        if col in cat_cols:
            print(pd.DataFrame({'COUNT': dataframe[col].value_counts(), 'RATIO': 100 * dataframe[col].value_counts(), 'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')
        if col in num_cols:
            print(pd.DataFrame({'MEAN': dataframe.groupby(target)[col].mean()}), end='\n\n\n')
target_summary(df, 'Outcome', cat_cols, num_cols)
sns.boxplot(x=df['Age'])


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)
outlier_thresholds(df, num_cols)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, num_cols)

def grab_outliers(dataframe, col_name, index=False):
    (low, up) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
    else:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
        return outlier_index
grab_outliers(df, 'Age', index=True)
grab_outliers(df, 'SkinThickness', index=True)
df.isnull().values.any()
df.isnull().sum()
df.notnull().sum()
df.isnull().sum().sum()
df[df.isnull().any(axis=1)]
df[df.notnull().all(axis=1)]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
missing_values_table(df, True)

def corr_plot(data, remove=['Id'], corr_coef='pearson', figsize=(20, 20)):
    if len(remove) > 0:
        num_cols2 = [x for x in data.columns if x not in remove]
    sns.set(font_scale=1.1)
    c = data[num_cols2].corr(method=corr_coef)
    mask = np.triu(c.corr(method=corr_coef))
    plt.figure(figsize=figsize)
    sns.heatmap(c, annot=True, fmt='.1f', cmap='coolwarm', square=True, mask=mask, linewidths=1, cbar=False)

corr_plot(df, corr_coef='spearman')
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.isnull().sum()
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=33)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()
corr_plot(df, corr_coef='spearman')
for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outlier(df, col))
df.head()
df['BMI'].max()
df['BMI_Ranges'] = pd.cut(df['BMI'], [0, 18.5, 25, 30, 45, 60], labels=['Underweight', 'Normal', 'Overweight', 'Fat', 'Obese'])

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)

cat_summary(df, 'BMI_Ranges', True)
df.groupby('Outcome')['BMI_Ranges'].value_counts()
cat_cols = ['BMI_Ranges']

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df[num_cols].head()
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)
from sklearn.ensemble import RandomForestClassifier