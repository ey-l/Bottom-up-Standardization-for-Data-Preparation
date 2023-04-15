import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
data.info()
data.describe().T
sns.countplot(x='Outcome', data=data)
data_copy = data.copy(deep=True)
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data_copy.isnull().sum()
p = data.hist(figsize=(20, 20))
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)
data_copy.isnull().sum()
import missingno as msno
p = msno.bar(data_copy)
p = sns.pairplot(data_copy, hue='Outcome')
plt.figure(figsize=(12, 10))
sns.heatmap(data_copy.corr(), annot=True, cmap='RdYlGn')
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
plt.figure(figsize=(6, 4))
sns.heatmap(data_copy.corr(), annot=True, cmap='RdYlGn')
ss = StandardScaler()
X = pd.DataFrame(ss.fit_transform(data_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = data_copy.Outcome
y
(train_x, test_x, train_y, test_y) = train_test_split(data_copy, y, test_size=0.3, random_state=42, stratify=y)
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)