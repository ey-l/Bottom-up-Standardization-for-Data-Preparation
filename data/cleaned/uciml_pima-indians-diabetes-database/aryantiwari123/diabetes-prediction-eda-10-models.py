import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.info()
data.shape
data.value_counts()
data.dtypes
data.columns
data.isnull().sum()
data.isnull().any()
data.isnull().all()
data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True)
data.hist(figsize=(18, 12))

plt.figure(figsize=(14, 10))
sns.set_style(style='whitegrid')
plt.subplot(2, 3, 1)
sns.boxplot(x='Glucose', data=data)
plt.subplot(2, 3, 2)
sns.boxplot(x='BloodPressure', data=data)
plt.subplot(2, 3, 3)
sns.boxplot(x='Insulin', data=data)
plt.subplot(2, 3, 4)
sns.boxplot(x='BMI', data=data)
plt.subplot(2, 3, 5)
sns.boxplot(x='Age', data=data)
plt.subplot(2, 3, 6)
sns.boxplot(x='SkinThickness', data=data)
mean_col = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'Outcome', 'BMI']
sns.pairplot(data[mean_col], palette='Accent')
sns.boxplot(x='Outcome', y='Insulin', data=data)
sns.regplot(x='BMI', y='Glucose', data=data)
sns.relplot(x='BMI', y='Glucose', data=data)
sns.scatterplot(x='Glucose', y='Insulin', data=data)
sns.jointplot(x='SkinThickness', y='Insulin', data=data)
sns.pairplot(data, hue='Outcome')
sns.lineplot(x='Glucose', y='Insulin', data=data)
sns.swarmplot(x='Glucose', y='Insulin', data=data)
sns.barplot(x='SkinThickness', y='Insulin', data=data[170:180])
plt.title('SkinThickness vs Insulin', fontsize=15)
plt.xlabel('SkinThickness')
plt.ylabel('Insulin')

plt.style.use('ggplot')
plt.style.use('default')
plt.figure(figsize=(5, 5))
sns.barplot(x='Glucose', y='Insulin', data=data[170:180])
plt.title('Glucose vs Insulin', fontsize=15)
plt.xlabel('Glucose')
plt.ylabel('Insulin')

x = data.drop(columns='Outcome')
y = data['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()