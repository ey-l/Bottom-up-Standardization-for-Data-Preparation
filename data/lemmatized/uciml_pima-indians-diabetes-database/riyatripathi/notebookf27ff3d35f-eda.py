import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
file = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
df_pima = pd.read_csv(file)
print('Rows: {} \nColumns: {}'.format(df_pima.shape[0], df_pima.shape[1]))
df_pima.describe()
print('Null value present in the dataset: ', df_pima.isnull().sum())
print('*************************************************************************')
df_pima.info()
print('*************************************************************************')
print('Duplicate Records: ', df_pima.duplicated().sum())
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_pima_copy = df_pima
for i in col:
    df_pima_copy[i].replace(to_replace=0, value=df_pima_copy[i].mean(), inplace=True)
df_pima_copy.describe()
(Q1, Q3) = np.percentile(df_pima_copy['Pregnancies'], [25, 95])
Q3
df_pima_copy['Pregnancies'] = np.where(df_pima_copy['Pregnancies'] > 10, 10, df_pima_copy['Pregnancies'])

def uiv(col):
    pass
    pass
    pass
for x in list(df_pima_copy.columns):
    uiv(x)
pass
fig.set_size_inches(15, 10)
pass
pass
pass
pass
pass
pass
pass
pass
pass
df_pima_copy.corr()
pass
mask = np.triu(np.ones_like(df_pima_copy.corr(), dtype=bool))
pass
pass