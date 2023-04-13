import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

def load():
    data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
    return data
df = load()
df.head()
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
num_cols = [col for col in df.columns if df[col].dtypes != 'O']
if len(cat_cols) == 0:
    print('There is not Categorical Column', ',', 'Number of Numerical Columns: ', len(num_cols), '\n', num_cols)
elif len(num_cols) == 0:
    print('There is not Numerical Column', ',', 'Number of Categorical Column: ', len(cat_cols), '\n', cat_cols)
else:
    print('')
num_cols = [col for col in df.columns if len(df[col].unique()) > 20 and df[col].dtypes != 'O' and (col not in 'Outcome')]
zero_columns = [col for col in df.columns if df[col].min() == 0 and col not in ['Pregnancies', 'Outcome']]
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])
df.describe([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
df.isnull().sum()
nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df.isnull().sum()

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    dtypes = dataframe.dtypes
    dtypesna = dtypes.loc[np.sum(dataframe.isnull()) != 0]
    missing_df = pd.concat([n_miss, np.round(ratio, 2), dtypesna], axis=1, keys=['n_miss', 'ratio', 'type'])
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print('no missing value')
missing_values_table(df)
for col in df.columns:
    df.loc[(df['Outcome'] == 0) & df[col].isnull(), col] = df[df['Outcome'] == 0][col].median()
    df.loc[(df['Outcome'] == 1) & df[col].isnull(), col] = df[df['Outcome'] == 1][col].median()
missing_values_table(df)
df['Glucose_Range'] = pd.cut(x=df['Glucose'], bins=[0, 140, 200], labels=['Normal', 'Prediabetes']).astype('O')
df['BMI_Range'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Healty', 'Overweight', 'Obese'])
df['BloodPressure_Range'] = pd.cut(x=df['BloodPressure'], bins=[0, 79, 89, 123], labels=['Normal', 'HS1', 'HS2'])
df['SkinThickness_Range'] = df['SkinThickness'].apply(lambda x: 1 if x <= 18.0 else 0)

def set_insulin(row):
    if row['Insulin'] >= 16 and row['Insulin'] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'
df['Insulin_range'] = df.apply(set_insulin, axis=1)
df.loc[df['Pregnancies'] == 0, 'NEW_PREG'] = 'NoPreg'
df.loc[(df['Pregnancies'] > 0) & (df['Pregnancies'] <= 4), 'NEW_PREG'] = 'NormalPreg'
df.loc[df['Pregnancies'] > 4, 'NEW_PREG'] = 'OverPreg'
df['BloodPres/Glucose'] = df['BloodPressure'] / df['Glucose']
df['Pregs*DiabetesPedigree'] = df['Pregnancies'] / df['DiabetesPedigreeFunction']
df['Pregs*DiabetesPedigree'] = df['Pregnancies'] * df['DiabetesPedigreeFunction']
df['BMI/Age'] = df['BMI'] / df['Age']
df.head()
from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']
for col in binary_cols:
    df = label_encoder(df, col)
ohe_cols = np.array([col for col in df.columns if 10 >= len(df[col].unique()) > 2])
df = one_hot_encoder(df, ohe_cols)
for col in df.columns:
    df.loc[(df['Outcome'] == 0) & df[col].isnull(), col] = df[df['Outcome'] == 0][col].median()
    df.loc[(df['Outcome'] == 1) & df[col].isnull(), col] = df[df['Outcome'] == 1][col].median()
df[df == np.inf] = np.nan
df.fillna(df.median(), inplace=True)
from sklearn.ensemble import RandomForestClassifier
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)