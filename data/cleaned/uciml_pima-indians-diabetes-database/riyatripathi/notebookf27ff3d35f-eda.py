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
    plt.figure(figsize=[9, 4])
    sns.boxplot(x=col, data=df_pima_copy, hue='Outcome')
    sns.displot(x=col, data=df_pima_copy, kde=True, color='g', hue='Outcome')

for x in list(df_pima_copy.columns):
    uiv(x)
(fig, axes) = plt.subplots(nrows=3, ncols=3)
fig.set_size_inches(15, 10)
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df_pima_copy, ax=axes[0][0])
sns.scatterplot(x='Glucose', y='Age', hue='Outcome', data=df_pima_copy, ax=axes[0][1])
sns.scatterplot(x='Glucose', y='Pregnancies', hue='Outcome', data=df_pima_copy, ax=axes[0][2])
sns.scatterplot(x='Glucose', y='BloodPressure', hue='Outcome', data=df_pima_copy, ax=axes[1][0])
sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=df_pima_copy, ax=axes[1][1])
sns.scatterplot(x='Glucose', y='Insulin', hue='Outcome', data=df_pima_copy, ax=axes[1][2])
sns.scatterplot(x='Age', y='Insulin', hue='Outcome', data=df_pima_copy, ax=axes[2][0])
sns.scatterplot(x='BMI', y='Insulin', hue='Outcome', data=df_pima_copy, ax=axes[2][1])
sns.scatterplot(x='BMI', y='DiabetesPedigreeFunction', hue='Outcome', data=df_pima_copy, ax=axes[2][2])

df_pima_copy.corr()
plt.figure(figsize=[12, 5])
mask = np.triu(np.ones_like(df_pima_copy.corr(), dtype=bool))
sns.heatmap(df_pima_copy.corr(), cmap='Blues', annot=True, mask=mask)

sns.pairplot(df_pima_copy, hue='Outcome', palette='bright')