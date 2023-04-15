import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
pima.info()
plt.figure(figsize=(8, 5))
sns.heatmap(pima.isnull(), yticklabels=False, cbar=False, cmap='viridis')
pima['Outcome'].value_counts()
sns.pairplot(pima, hue='Outcome')
pima.describe()
corrmat = pima.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corrmat, vmax=0.8, square=True, annot=True)
pima.columns
pima.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
(f, axes) = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
sns.distplot(pima.Pregnancies, kde=False, color='g', ax=axes[0][0]).set_title('Pregnanices')
axes[0][0].set_ylabel('Count')
sns.distplot(pima.Glucose, kde=False, color='r', ax=axes[0][1]).set_title('Glucose')
axes[0][1].set_ylabel('Count')
sns.distplot(pima.BloodPressure, kde=False, color='b', ax=axes[1][0]).set_title('Blood Pressure')
axes[1][0].set_ylabel('Count')
sns.distplot(pima.SkinThickness, kde=False, color='g', ax=axes[1][1]).set_title('Skin Thickness')
axes[1][1].set_ylabel('Count')
sns.distplot(pima.Insulin, kde=False, color='r', ax=axes[2][0]).set_title('Insulin')
axes[2][0].set_ylabel('Count')
sns.distplot(pima.BMI, kde=False, color='b', ax=axes[2][1]).set_title('BMI')
axes[2][1].set_ylabel('Count')
sns.distplot(pima.DiabetesPedigreeFunction, kde=False, color='g', ax=axes[3][0]).set_title('DiabetesPedigreeFunction')
axes[3][0].set_ylabel('Count')
sns.distplot(pima.Age, kde=False, color='r', ax=axes[3][1]).set_title('Age')
axes[3][1].set_ylabel('Count')
pima_new = pima
pima_new.info()
pima_new = pima_new[pima_new['Pregnancies'] < 13]
pima_new = pima_new[pima_new['Glucose'] > 30]
pima_new = pima_new[pima_new['BMI'] > 10]
pima_new = pima_new[pima_new['BMI'] < 50]
pima_new = pima_new[pima_new['DiabetesPedigreeFunction'] < 1.2]
pima_new = pima_new[pima_new['Age'] < 65]
pima_new.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(10, 8))
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 0, 'Pregnancies'], color='b', shade=True, label='No')
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 1, 'Pregnancies'], color='r', shade=True, label='Yes')
ax.set(xlabel='Pregnancies', ylabel='Frequency')
plt.title('Pregnancies vs Yes or No')
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 0, 'Glucose'], color='b', shade=True, label='No')
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 1, 'Glucose'], color='r', shade=True, label='Yes')
ax.set(xlabel='Glucose', ylabel='Frequency')
plt.title('Glucose vs Yes or No')
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 0, 'BMI'], color='b', shade=True, label='No')
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 1, 'BMI'], color='r', shade=True, label='Yes')
ax.set(xlabel='BMI', ylabel='Frequency')
plt.title('BMI vs Yes or No')
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 0, 'DiabetesPedigreeFunction'], color='b', shade=True, label='No')
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 1, 'DiabetesPedigreeFunction'], color='r', shade=True, label='Yes')
ax.set(xlabel='DiabetesPedigreeFunction', ylabel='Frequency')
plt.title('DiabetesPedigreeFunction vs Yes or No')
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 0, 'Age'], color='b', shade=True, label='No')
ax = sns.kdeplot(pima_new.loc[pima_new['Outcome'] == 1, 'Age'], color='r', shade=True, label='Yes')
ax.set(xlabel='Age', ylabel='Frequency')
plt.title('Age vs Yes or No')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(pima_new.drop('Outcome', axis=1), pima_new['Outcome'], test_size=0.3, random_state=123)
from sklearn import preprocessing