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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe([0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).T
df.groupby('Outcome').agg('mean')

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)
(low, up) = outlier_thresholds(df, 'Pregnancies')

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, 'Pregnancies')
check_outlier(df, 'BloodPressure')
check_outlier(df, 'SkinThickness')
check_outlier(df, 'Insulin')
check_outlier(df, 'BMI')
check_outlier(df, 'DiabetesPedigreeFunction')
check_outlier(df, 'Age')
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')

th = np.sort(df_scores)[10]
df[df_scores < th]
df[df_scores < th].shape
df.isnull().sum()
zero_columns = [i for i in df.columns if df[i].min() == 0 and i not in ['Pregnancies', 'Outcome']]
for i in zero_columns:
    df[[i]] = df[[i]].replace(0, np.NaN)
df.isnull().sum()
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()
df.isnull().sum()

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
replace_with_thresholds(df, 'Pregnancies')
replace_with_thresholds(df, 'BloodPressure')
replace_with_thresholds(df, 'SkinThickness')
replace_with_thresholds(df, 'Insulin')
replace_with_thresholds(df, 'BMI')
replace_with_thresholds(df, 'DiabetesPedigreeFunction')
replace_with_thresholds(df, 'Age')
check_outlier(df, 'Pregnancies')
check_outlier(df, 'BloodPressure')
check_outlier(df, 'SkinThickness')
check_outlier(df, 'Insulin')
check_outlier(df, 'BMI')
check_outlier(df, 'DiabetesPedigreeFunction')
check_outlier(df, 'Age')
df.corrwith(df['Outcome']).sort_values(ascending=False)
corr_df = df.corr()
sns.heatmap(corr_df, annot=True, xticklabels=corr_df.columns, yticklabels=corr_df.columns)

df['Age-Insul'] = df['Age'] * df['Insulin']
df['Age-BMI'] = df['Age'] * df['BMI']
df['Preg-Insul'] = df['Pregnancies'] * df['Insulin']
df['Gluc-Insul'] = df['Glucose'] * df['Insulin']
df['SkinT-Age'] = df['SkinThickness'] * df['Age']
df['Preg-SkinT'] = df['SkinThickness'] * df['Pregnancies']

def new_insulin(row):
    if row['BloodPressure'] > 80:
        return 'Hipertansiyon'
    elif row['BloodPressure'] < 60:
        return 'Hipotansiyon'
    else:
        return 'Normal'
df = df.assign(NewInsulin=df.apply(new_insulin, axis=1))

def new_age(row):
    if row['Age'] > 40:
        return 'Olgunyaş'
    elif row['Age'] <= 40 and row['Age'] > 30:
        return 'Ortayaş'
    elif row['Age'] <= 30 and row['Age'] >= 25:
        return 'Gençyaş'
    else:
        return 'Ergen'
df = df.assign(NewAge=df.apply(new_age, axis=1))

def new_bmı(row):
    if row['BMI'] > 40:
        return 'Morbidobez'
    elif row['BMI'] <= 40 and row['BMI'] > 35:
        return 'Tip2obez'
    elif row['BMI'] <= 35 and row['BMI'] > 30:
        return 'Tip1obez'
    elif row['BMI'] <= 30 and row['BMI'] > 25:
        return 'Fazlakilolu'
    else:
        return 'Normal'
df = df.assign(NewBMI=df.apply(new_bmı, axis=1))

def new_glucose(row):
    if row['Glucose'] < 70:
        return 'Düşük'
    elif row['Glucose'] < 100 and row['Glucose'] >= 70:
        return 'Normal'
    elif row['Glucose'] < 125 and row['Glucose'] >= 100:
        return 'Potansiyel'
    else:
        return 'Yüksek'
df = df.assign(NewGlucose=df.apply(new_glucose, axis=1))
df.head()
df = pd.get_dummies(df, columns=['NewGlucose'], drop_first=True)
df = pd.get_dummies(df, columns=['NewBMI'], drop_first=True)
df = pd.get_dummies(df, columns=['NewAge'], drop_first=True)
df = pd.get_dummies(df, columns=['NewInsulin'], drop_first=True)
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)