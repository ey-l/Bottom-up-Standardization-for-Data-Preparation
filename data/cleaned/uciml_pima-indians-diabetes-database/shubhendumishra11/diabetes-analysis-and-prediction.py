import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
data.isnull().sum()
sns.countplot(data['Pregnancies'])
sns.distplot(data['Glucose'])
sns.distplot(data['BloodPressure'])
sns.distplot(data['SkinThickness'])
sns.distplot(data['Insulin'])
sns.distplot(data['BMI'])
sns.distplot(data['DiabetesPedigreeFunction'])
sns.distplot(data['Age'])
sns.countplot(data['Outcome'])
sns.swarmplot(x='Outcome', y='Pregnancies', data=data)
sns.swarmplot(x='Outcome', y='Age', data=data)
sns.lmplot(x='Insulin', y='Glucose', hue='Outcome', data=data)
sns.lmplot(x='DiabetesPedigreeFunction', y='Glucose', hue='Outcome', data=data)
sns.lmplot(x='Pregnancies', y='Glucose', hue='Outcome', data=data)
sns.lmplot(x='BloodPressure', y='Glucose', hue='Outcome', data=data)
sns.lmplot(x='SkinThickness', y='BMI', hue='Outcome', data=data)
sns.lmplot(x='SkinThickness', y='Insulin', hue='Outcome', data=data)
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(method='pearson'), annot=True)
y = data['Outcome']
X = data.drop(columns=['Outcome'], inplace=True)
features = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
X = data[features]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, shuffle=True)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()