import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
dataset.info()
dataset.isnull().sum()
dataset.describe()
dataset.Outcome.value_counts()
sns.countplot(x='Outcome', data=dataset)
print(sns.distplot(dataset['Pregnancies']))
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.boxplot(data=dataset['Pregnancies'], ax=axes[0])
sns.violinplot(data=dataset['Pregnancies'], ax=axes[1])
sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(sns.distplot, 'Pregnancies').add_legend()
plt.title('PDF with Pregnancies')

sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'Pregnancies').add_legend()
plt.title('Sebaran Pasien Berdasarkan Pregnancies')

print(sns.distplot(dataset['Glucose']))
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.boxplot(data=dataset['Glucose'], ax=axes[0])
sns.violinplot(data=dataset['Glucose'], ax=axes[1])
sns.FacetGrid(dataset, hue='Outcome', height=5).map(sns.distplot, 'Glucose').add_legend()
plt.title('PDF with Glucose')

sns.FacetGrid(dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'Glucose').add_legend()
plt.title('Distribusi Pasien Berdasarkan Glucose')

sns.distplot(dataset['BloodPressure'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.boxplot(data=dataset['BloodPressure'], ax=axes[0])
sns.violinplot(data=dataset['BloodPressure'], ax=axes[1])
sns.FacetGrid(dataset, hue='Outcome', height=5).map(sns.distplot, 'BloodPressure').add_legend()
plt.title('PDF with BloodPressure')

sns.FacetGrid(dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'BloodPressure').add_legend()
plt.title('Distribusi Pasien Berdasarkan BloodPressure')

sns.distplot(dataset['SkinThickness'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.boxplot(data=dataset['SkinThickness'], ax=axes[0])
sns.violinplot(data=dataset['SkinThickness'], ax=axes[1])
sns.FacetGrid(dataset, hue='Outcome', height=5).map(sns.distplot, 'SkinThickness').add_legend()
plt.title('PDF with SkinThickness')

sns.FacetGrid(dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'SkinThickness').add_legend()
plt.title('Distribusi Pasien Berdasarkan SkinThickness')

sns.distplot(dataset['Insulin'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.boxplot(data=dataset['Insulin'], ax=axes[0])
sns.violinplot(data=dataset['Insulin'], ax=axes[1])
sns.FacetGrid(dataset, hue='Outcome', height=5).map(sns.distplot, 'Insulin').add_legend()
plt.title('PDF with Insulin')

sns.FacetGrid(dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'Insulin').add_legend()
plt.title('Distribusi Pasien Berdasarkan Insulin')

sns.distplot(dataset['BMI'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
sns.barplot(data=dataset['BMI'], ax=axes[0])
sns.violinplot(data=dataset['BMI'], ax=axes[1])
sns.FacetGrid(dataset, hue='Outcome', height=5).map(sns.distplot, 'BMI').add_legend()
plt.title('PDF with BMI')

sns.FacetGrid(dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'BMI').add_legend()
plt.title('Sebaran pasien berdasarkan BMI')

sns.distplot(dataset['DiabetesPedigreeFunction'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
sns.boxplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[0])
sns.violinplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[1])
sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(sns.distplot, 'DiabetesPedigreeFunction').add_legend()
plt.title('PDF with DiabetesPedigreeFunction')

sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'DiabetesPedigreeFunction').add_legend()
plt.title('Sebaran pasien berdasarkan DiabetesPedigreeFunction')

sns.distplot(dataset['Age'])
(_, axes) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
sns.boxplot(data=dataset['Age'], ax=axes[0])
sns.violinplot(data=dataset['Age'], ax=axes[1])
sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(sns.distplot, 'Age').add_legend()
plt.title('PDF with Age')

sns.FacetGrid(data=dataset, hue='Outcome', height=5).map(plt.scatter, 'Outcome', 'Age').add_legend()
plt.title('Sebaran pasien berdasarkan Age')

sns.pairplot(data=dataset)
from statsmodels.tools import add_constant as add_constant
dataset_df = add_constant(dataset)
dataset_df.head()
import statsmodels.api as sm
column = dataset_df.columns[:-1]
model = sm.Logit(dataset_df.Outcome, dataset_df[column])