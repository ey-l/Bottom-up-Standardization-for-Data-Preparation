import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.shape
diabetes.head()
diabetes.tail()
diabetes = diabetes.rename(columns={'BloodPressure': 'Blood_Pressure', 'SkinThickness': 'Skin_Thickness', 'DiabetesPedigreeFunction': 'Diabetes_Pedigree_Function'})
diabetes.info()
diabetes['Pregnancies'].unique()
diabetes['Glucose'].unique()
diabetes['Blood_Pressure'].unique()
diabetes['Skin_Thickness'].unique()
diabetes['Insulin'].unique()
diabetes['BMI'].unique()
diabetes['Diabetes_Pedigree_Function'].unique()
diabetes['Age'].unique()
diabetes['Outcome'].unique()
diabetes_missing = diabetes.replace({'Glucose': {0: np.nan}, 'Blood_Pressure': {0: np.nan}, 'Skin_Thickness': {0: np.nan}, 'Insulin': {0: np.nan}, 'BMI': {0: np.nan}})
diabetes_missing.isna().mean() * 100
from fancyimpute import KNN
knn_imputer = KNN()
diabetes_knn = diabetes_missing.copy(deep=True)
diabetes_knn.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn)
diabetes_knn.isna().sum()
diabetes_knn.duplicated().sum()
diabetes_knn.describe()
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
df = diabetes_knn.copy()
low_perc_insulin = 0.1
high_perc_insulin = 0.9
(low_insulin, upper_insulin) = df.Insulin.quantile([low_perc_insulin, high_perc_insulin])
print(low_insulin)
print(upper_insulin)
df = df[~((df.Insulin > 272) | (df.Insulin < 56))]
low_perc_bp = 0.1
high_perc_bp = 0.9
(low_bp, upper_bp) = df.Blood_Pressure.quantile([low_perc_bp, high_perc_bp])
print(low_bp)
print(upper_bp)
df = df[~((df.Blood_Pressure > 88) | (df.Blood_Pressure < 58))]
low_perc_age = 0.1
high_perc_age = 0.9
(low_age, upper_age) = df.Age.quantile([low_perc_age, high_perc_age])
print(low_age)
print(upper_age)
df = df[~((df.Age > 50) | (df.Age < 22))]
low_perc_BMI = 0.1
high_perc_BMI = 0.9
(low_BMI, upper_BMI) = df.BMI.quantile([low_perc_BMI, high_perc_BMI])
print(low_BMI)
print(upper_BMI)
df = df[~((df.BMI > 41) | (df.BMI < 26))]
low_perc_st = 0.1
high_perc_st = 0.9
(low_st, upper_st) = df.Skin_Thickness.quantile([low_perc_st, high_perc_st])
print(low_st)
print(upper_st)
df = df[~((df.Skin_Thickness > 40) | (df.Skin_Thickness < 19))]
low_perc_preg = 0.1
high_perc_preg = 0.9
(low_preg, upper_preg) = df.Pregnancies.quantile([low_perc_preg, high_perc_preg])
print(low_preg)
print(upper_preg)
df = df[~((df.Pregnancies > 9) | (df.Pregnancies < 0))]
pass
labels = [0, 1]
values = df['Outcome'].value_counts()
pass
df.groupby('Outcome')['Skin_Thickness'].mean()
df.groupby('Outcome')['Skin_Thickness'].mean().plot.bar()
px.box(df, y='Skin_Thickness', color='Outcome', width=800, height=400)
px.box(df, y='Pregnancies', color='Outcome', width=800, height=400)
px.box(df, y='Insulin', color='Outcome', width=700, height=350)
px.box(df, y='Glucose', color='Outcome', width=700, height=350)
df.groupby('Outcome')['Blood_Pressure'].mean().plot.bar()
px.box(df, y='Blood_Pressure', x='Outcome', width=800, height=400)
df.groupby('Outcome')['BMI'].mean().plot.bar()
px.box(df, y='BMI', x='Outcome', width=800, height=400)
px.box(df, y='Age', color='Outcome', width=800, height=400)
px.scatter(diabetes_knn, y='BMI', x='Age', color='Outcome', title='BMI and Age vs Outcome', width=800, height=400)
px.scatter(diabetes_knn, y='BMI', x='Glucose', color='Outcome', title='BMI and Glucose vs Outcome', width=800, height=400)
px.scatter(diabetes_knn, y='Insulin', x='Age', color='Outcome', title='Insulin and Age vs Outcome', width=800, height=400)
px.scatter(diabetes_knn, y='Glucose', x='Age', color='Outcome', title='Glucose and Age vs Outcome', width=800, height=400)
px.scatter(diabetes_knn, y='Blood_Pressure', x='Age', color='Outcome', title='Blood Pressure and Age vs Outcome', width=800, height=400)