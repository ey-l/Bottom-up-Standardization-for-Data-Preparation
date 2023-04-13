import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(10)
df.sample(10)
df.tail(10)
df.info()
df.columns
df.shape
df.isnull().sum()
df.describe().T
df.corr()
df_yeni = df.copy(deep=True)
df_yeni[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_yeni[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df_yeni.sample(10)
df_yeni.isnull().sum()
df_yeni.describe().T
df_yeni.corr()
df.hist(figsize=(20, 20))
df_yeni.hist(figsize=(20, 20))
df_yeni['Glucose'].fillna(df_yeni['Glucose'].median(), inplace=True)
df_yeni['BloodPressure'].fillna(df_yeni['BloodPressure'].median(), inplace=True)
df_yeni['SkinThickness'].fillna(df_yeni['SkinThickness'].median(), inplace=True)
df_yeni['Insulin'].fillna(df_yeni['Insulin'].median(), inplace=True)
df_yeni['BMI'].fillna(df_yeni['BMI'].median(), inplace=True)
df_yeni.isnull().sum()
df_yeni.describe().T
df_yeni.corr()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
df_yeni['Outcome'].value_counts()
pass
pass
labels = ('Diabetic', 'Healthy')
df_yeni.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
df_yeni['Age'] = df_yeni['Age']
bins = [20, 35, 50, 65, 81]
labels = ['Genç', 'Orta Yaş', 'Yetişkin', 'Yaşlı']
df_yeni['yas_grp'] = pd.cut(df_yeni['Age'], bins, labels=labels)
df_yeni.head()
df_yeni.yas_grp.value_counts()
colors = ['green', 'yellow', 'orange', 'red']
labels = df_yeni.yas_grp.value_counts().index
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
y = df_yeni['Outcome'].values
x_data = df_yeni.drop(['Outcome', 'yas_grp'], axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x.head()
import statsmodels.api as sm