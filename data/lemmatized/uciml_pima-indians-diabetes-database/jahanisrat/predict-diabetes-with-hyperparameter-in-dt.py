import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
import itertools
pass
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('data/input/uciml_pima-indians-diabetes-database'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(diabetes.columns)
diabetes.shape
diabetes.head()
diabetes['Outcome'].value_counts()
print(diabetes.isnull().values.any())
0 in diabetes.values
print('# rows in dataframe {0}'.format(len(diabetes)))
print('Zero in Pregnancies : {0}'.format(len(diabetes.loc[diabetes['Pregnancies'] == 0])))
print('Zero in Glucose : {0}'.format(len(diabetes.loc[diabetes['Glucose'] == 0])))
print('Zero in BloodPressure: {0}'.format(len(diabetes.loc[diabetes['BloodPressure'] == 0])))
print('Zero in SkinThickness : {0}'.format(len(diabetes.loc[diabetes['SkinThickness'] == 0])))
print('Zero in Insulin  : {0}'.format(len(diabetes.loc[diabetes['Insulin'] == 0])))
print('Zero in BMI : {0}'.format(len(diabetes.loc[diabetes['BMI'] == 0])))
print('Zero in DiabetesPedigreeFunction  : {0}'.format(len(diabetes.loc[diabetes['DiabetesPedigreeFunction'] == 0])))
print('Zero in Age: {0}'.format(len(diabetes.loc[diabetes['Age'] == 0])))
R_d = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
R_d.head()
R_d.isnull().sum()[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
pd.options.display.float_format = '{:,.2f}'.format
diabetes['Glucose'].fillna(diabetes['Glucose'].mean(), inplace=True)
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(), inplace=True)
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].mean(), inplace=True)
diabetes['Insulin'].fillna(diabetes['Insulin'].mean(), inplace=True)
diabetes['BMI'].fillna(diabetes['BMI'].mean(), inplace=True)
diabetes.head()
diabetes['Outcome'].value_counts()
pass
pass
labels = ('Healthy', 'Diabetic')
diabetes.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
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
pd.options.display.float_format = '{:,.3f}'.format
correlation = diabetes.corr()
correlation
pass
pass
pass
(bottom, top) = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
pass
Feature = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X = Feature
X[0:5]
y = diabetes['Outcome'].values
y[0:5]
from sklearn import preprocessing