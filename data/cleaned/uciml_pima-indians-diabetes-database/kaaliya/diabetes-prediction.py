import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data
diabetes_data.info()
plt.figure(figsize=(15, 15))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(diabetes_data.corr(), annot=True, cmap=cmap)
exists = 0 in diabetes_data.Glucose
print(exists)
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for cols in col:
    diabetes_data[cols].replace(0, diabetes_data[cols].mean(), inplace=True)
diabetes_data.hist(figsize=(20, 20))
sns.pointplot(x='Pregnancies', y='Age', data=diabetes_data)
sns.pointplot(x='Outcome', y='Glucose', data=diabetes_data)
plt.figure(figsize=(20, 20))
sns.pointplot(x='SkinThickness', y='Insulin', data=diabetes_data)
sns.regplot(x='SkinThickness', y='Insulin', data=diabetes_data)
sns.regplot(x='Pregnancies', y='Age', data=diabetes_data)
sns.pairplot(data=diabetes_data, hue='Outcome')
sns.swarmplot(x='Pregnancies', y='Age', data=diabetes_data, hue='Outcome')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = diabetes_data.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.5, random_state=3)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()