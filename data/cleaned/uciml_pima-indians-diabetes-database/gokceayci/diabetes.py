import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
df.isnull().values.any()
df.eq(0).sum()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
df.fillna(df.mean(), inplace=True)
df.head()
df.isnull().sum()
df.eq(0).sum()
p = df.hist(bins=50, figsize=(20, 15))

df.plot(kind='density', subplots=True, layout=(3, 3), figsize=(20, 15), sharex=False)

plt.figure(figsize=(25, 15))
sns.pairplot(df)

sns.pairplot(df, hue='Outcome')
plt.figure(figsize=(15, 20))
plt.subplot(4, 2, 1)
sns.violinplot(x='Outcome', y='Pregnancies', data=df, palette='Set2')
plt.subplot(4, 2, 2)
sns.violinplot(x='Outcome', y='Glucose', data=df, palette='Set2')
plt.subplot(4, 2, 3)
sns.violinplot(x='Outcome', y='BloodPressure', data=df, palette='Set2')
plt.subplot(4, 2, 4)
sns.violinplot(x='Outcome', y='SkinThickness', data=df, palette='Set2')
plt.subplot(4, 2, 5)
sns.violinplot(x='Outcome', y='Insulin', data=df, palette='Set2')
plt.subplot(4, 2, 6)
sns.violinplot(x='Outcome', y='BMI', data=df, palette='Set2')
plt.subplot(4, 2, 7)
sns.violinplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, palette='Set2')
plt.subplot(4, 2, 8)
sns.violinplot(x='Outcome', y='Age', data=df, palette='Set2')
plt.figure(figsize=(15, 20))
plt.subplot(4, 2, 1)
sns.boxplot(x='Outcome', y='Pregnancies', data=df, palette='Set2')
plt.subplot(4, 2, 2)
sns.boxplot(x='Outcome', y='Glucose', data=df, palette='Set2')
plt.subplot(4, 2, 3)
sns.boxplot(x='Outcome', y='BloodPressure', data=df, palette='Set2')
plt.subplot(4, 2, 4)
sns.boxplot(x='Outcome', y='SkinThickness', data=df, palette='Set2')
plt.subplot(4, 2, 5)
sns.boxplot(x='Outcome', y='Insulin', data=df, palette='Set2')
plt.subplot(4, 2, 6)
sns.boxplot(x='Outcome', y='BMI', data=df, palette='Set2')
plt.subplot(4, 2, 7)
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, palette='Set2')
plt.subplot(4, 2, 8)
sns.boxplot(x='Outcome', y='Age', data=df, palette='Set2')
c = ['green', 'red']
print(df.Outcome.value_counts())
df.Outcome.value_counts().plot(kind='bar', color=c)
df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')

df.corr().nlargest(4, 'Outcome').index
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
x = df[['Glucose', 'BMI', 'Age']]
y = df.iloc[:, 8]
y
log_reg = linear_model.LogisticRegression()
log_reg_score = cross_val_score(log_reg, x, y, cv=10, scoring='accuracy').mean()
log_reg_score
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svem_score = cross_val_score(linear_svm, x, y, cv=10, scoring='accuracy').mean()
linear_svem_score
import pickle
filename = 'diabetes.sav'