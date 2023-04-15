import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

import missingno as msno
import sklearn as sk
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Number of row and columns in dataset', '-' * 130)

print('First five row of dataset', '-' * 145)

print('Last five row of dataset', '-' * 145)

print('data type  of each values', '-' * 145)

print('Null values in dataset', '-' * 145)

diabetes = diabetes.astype({'Outcome': 'category'})
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
diabetes[cols] = diabetes[cols].replace({'0': np.nan, 0: np.nan})
diabetes.isnull().sum()
msno.bar(diabetes)
diabetes.describe().round().T
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['Age'], kde=True, color='red', bins=30)
plt.title('Age Distribution', fontsize=18)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.Age.mean(), color='green', label='mean')
plt.axvline(x=diabetes.Age.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['DiabetesPedigreeFunction'], kde=True, color='red', bins=50)
plt.title('DiabetesPedigreeFunction Distribution', fontsize=18)
plt.xlabel('DiabetesPedigreeFunction', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.DiabetesPedigreeFunction.mean(), color='green', label='mean')
plt.axvline(x=diabetes.DiabetesPedigreeFunction.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['Insulin'], kde=True, color='red', bins=50)
plt.title('Insulin Distribution', fontsize=18)
plt.xlabel('Insulin', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.Insulin.mean(), color='green', label='mean')
plt.axvline(x=diabetes.Insulin.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['SkinThickness'], kde=True, color='red', bins=50)
plt.title('SkinThickness Distribution', fontsize=18)
plt.xlabel('SkinThickness', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.SkinThickness.mean(), color='green', label='mean')
plt.axvline(x=diabetes.SkinThickness.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['BloodPressure'], kde=True, color='red', bins=50)
plt.title('BloodPressure Distribution', fontsize=18)
plt.xlabel('BloodPressure', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.BloodPressure.mean(), color='green', label='mean')
plt.axvline(x=diabetes.BloodPressure.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['Glucose'], kde=True, color='red', bins=50)
plt.title('Glucose Distribution', fontsize=18)
plt.xlabel('Glucose', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.Glucose.mean(), color='green', label='mean')
plt.axvline(x=diabetes.Glucose.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['BMI'], kde=True, color='red', bins=50)
plt.title('BMI Distribution', fontsize=18)
plt.xlabel('BMI', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.BMI.mean(), color='green', label='mean')
plt.axvline(x=diabetes.BMI.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set_theme(style='whitegrid')
(fig, ax) = plt.subplots(figsize=(24, 24), nrows=3, ncols=3)
sns.boxplot(data=diabetes, y='Pregnancies', ax=ax[0, 0], color='red')
sns.boxplot(data=diabetes, y='Glucose', ax=ax[0, 1], color='red')
sns.boxplot(data=diabetes, y='BloodPressure', ax=ax[0, 2], color='red')
sns.boxplot(data=diabetes, y='SkinThickness', ax=ax[1, 0], color='red')
sns.boxplot(data=diabetes, y='Insulin', ax=ax[1, 1], color='red')
sns.boxplot(data=diabetes, y='Age', ax=ax[2, 1], color='red')
sns.boxplot(data=diabetes, y='BMI', ax=ax[2, 0], color='red')
sns.boxplot(data=diabetes, y='DiabetesPedigreeFunction', ax=ax[1, 2], color='red')
hm = sns.heatmap(diabetes.corr(), annot=True)
hm.set(title='Correlation matrix')

diabetes['SkinThickness'].fillna(int(diabetes['SkinThickness'].median()), inplace=True)
diabetes['Insulin'].fillna(int(diabetes['Insulin'].median()), inplace=True)
diabetes.loc[(diabetes['Age'] >= 21) & (diabetes['Age'] < 50), 'age_group'] = 'Adult'
diabetes.loc[diabetes['Age'] >= 50, 'age_group'] = 'Senior'
diabetes['Glucose_Range'] = pd.cut(x=diabetes['Glucose'], bins=[0, 140, 200, 300], labels=['Normal', 'Prediabetes', 'Diabetes'])
diabetes['BMI_Group'] = pd.cut(x=diabetes['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
diabetes['SkinThickness_log'] = np.log(diabetes['SkinThickness'])
diabetes['DiabetesPedigreeFunction_log'] = np.log(diabetes['DiabetesPedigreeFunction'])
diabetes['Insulin_log'] = np.log(diabetes['Insulin'])
sugar = diabetes[['age_group', 'BMI_Group', 'BloodPressure', 'SkinThickness_log', 'Glucose_Range', 'Insulin_log', 'Pregnancies', 'DiabetesPedigreeFunction_log', 'Outcome']].copy()
X = sugar.drop(columns='Outcome', axis=1)
X_f = pd.get_dummies(X)
X_f
num = ['BloodPressure', 'Pregnancies']
scaler = RobustScaler()
X_f[num] = scaler.fit_transform(X_f[num])
X_f
high = X_f.drop(['BloodPressure'], axis=1)
high
X_f.isnull().sum()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['SkinThickness_log'], kde=True, color='red', bins=30)
plt.title('SkinThickness_log Distribution', fontsize=18)
plt.xlabel('SkinThickness_log', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.SkinThickness_log.mean(), color='green', label='mean')
plt.axvline(x=diabetes.SkinThickness_log.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_theme(style='whitegrid')
sns.histplot(diabetes['Insulin_log'], kde=True, color='red', bins=20)
plt.title('Insulin_log Distribution', fontsize=18)
plt.xlabel('Insulin_log', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.axvline(x=diabetes.Insulin_log.mean(), color='green', label='mean')
plt.axvline(x=diabetes.Insulin_log.median(), color='blue', ls='--', lw=2.5, label='medain')
plt.legend()
sugar = diabetes[['age_group', 'BMI_Group', 'BloodPressure', 'SkinThickness_log', 'Glucose_Range', 'Insulin_log', 'Pregnancies', 'DiabetesPedigreeFunction_log', 'Outcome']]
diabetes['Outcome'].value_counts().plot(kind='pie', figsize=(10, 10))
diabetes['Glucose_Range'].value_counts().plot(kind='pie', figsize=(10, 10))
diabetes['BMI_Group'].value_counts().plot(kind='pie', figsize=(10, 10))
diabetes['age_group'].value_counts().plot(kind='pie', figsize=(10, 10))
Y = diabetes['Outcome']
labels = Y
features = X_f
(train_features, test_features, train_labels, test_labels) = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=2)
train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
test_features.replace([np.inf, -np.inf], np.nan, inplace=True)
train_features.fillna(999, inplace=True)
test_features.fillna(999, inplace=True)
model = LogisticRegression(solver='liblinear')