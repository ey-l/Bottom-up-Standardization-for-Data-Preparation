import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.head(12)
df.shape
df.columns
df.info()
df.describe().T
df.nunique()
df['Outcome'].value_counts()
df['Outcome'].value_counts() * 100 / len(df)
df.groupby('Outcome').agg({'Pregnancies': 'mean'})
df.groupby('Outcome').agg({'Age': 'mean'})
df.groupby('Outcome').agg({'Age': 'max'})
df.groupby('Outcome').agg({'Insulin': 'mean'})
df.groupby('Outcome').agg({'Insulin': 'max'})
df.groupby('Outcome').agg({'Glucose': 'mean'})
df.groupby('Outcome').agg({'Glucose': 'max'})
df.groupby('Outcome').agg({'BMI': 'mean'})
df.duplicated().sum()
df.isna().sum()
(df[df.columns] == 0).sum()
for i in ['Glucose', 'BMI', 'Insulin', 'BloodPressure']:
    df[i].replace({0: df[i].median()}, inplace=True)
(df[df.columns] == 0).sum()
plt.boxplot(df)

def outlier_treatment():
    l = ['BMI', 'Glucose', 'SkinThickness', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies', 'DiabetesPedigreeFunction']
    for i in l:
        x = np.quantile(df[i], [0.25, 0.75])
        iqr = x[1] - x[0]
        uw = x[1] + 1.5 * iqr
        lw = x[0] - 1.5 * iqr
        df[i] = np.where(df[i] > uw, uw, np.where(df[i] < lw, lw, df[i]))
outlier_treatment()
plt.boxplot(df)
df.hist(bins=50, figsize=(20, 15))
df['Outcome'].hist()
df['Age'].hist()
df['Pregnancies'].hist()
df.corr()
df.corr()['Outcome'].sort_values(ascending=False)
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.title('correlation of feature')
sns.pairplot(df, hue='Outcome', palette='husl')
pno = 1
plt.figure(figsize=(18, 20))
for i in df.columns:
    if pno < 9:
        plt.subplot(3, 3, pno)
        ax = sns.histplot(data=df, x=i, hue=df.Outcome, kde=True)
        plt.xlabel(i)
        pno += 1
        for i in ax.containers:
            ax.bar_label(i)
sns.countplot(x='Outcome', data=df)
plt.title('Outcome')
plt.pie(df['Outcome'].value_counts(), labels=['No', 'Yes'], autopct='%1.2f%%')

NewBMI = pd.Series(['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2', 'Obesity 3'], dtype='category')
df['NewBMI'] = NewBMI
df.loc[df['BMI'] < 18.5, 'NewBMI'] = NewBMI[0]
df.loc[(df['BMI'] > 18.5) & (df['BMI'] <= 24.9), 'NewBMI'] = NewBMI[1]
df.loc[(df['BMI'] > 24.9) & (df['BMI'] <= 29.9), 'NewBMI'] = NewBMI[2]
df.loc[(df['BMI'] > 29.9) & (df['BMI'] <= 34.9), 'NewBMI'] = NewBMI[3]
df.loc[(df['BMI'] > 34.9) & (df['BMI'] <= 39.9), 'NewBMI'] = NewBMI[4]
df.loc[df['BMI'] > 39.9, 'NewBMI'] = NewBMI[5]
df.head()

def set_insulin(row):
    if row['Insulin'] >= 16 and row['Insulin'] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
df.head()
NewGlucose = pd.Series(['Low', 'Normal', 'Overweight', 'Secret', 'High'], dtype='category')
df['NewGlucose'] = NewGlucose
df.loc[df['Glucose'] <= 70, 'NewGlucose'] = NewGlucose[0]
df.loc[(df['Glucose'] > 70) & (df['Glucose'] <= 99), 'NewGlucose'] = NewGlucose[1]
df.loc[(df['Glucose'] > 99) & (df['Glucose'] <= 126), 'NewGlucose'] = NewGlucose[2]
df.loc[df['Glucose'] > 126, 'NewGlucose'] = NewGlucose[3]
df.head()
df = pd.get_dummies(df, columns=['NewBMI', 'NewInsulinScore', 'NewGlucose'], drop_first=True)
df.head()
df.head()
x = df.drop(columns=['Outcome'])
y = df['Outcome']
cols = x.columns
index = x.index
from sklearn.preprocessing import RobustScaler