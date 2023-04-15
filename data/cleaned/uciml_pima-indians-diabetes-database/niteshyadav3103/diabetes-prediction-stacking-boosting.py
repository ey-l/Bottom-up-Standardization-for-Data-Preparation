import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
plt.figure(figsize=(20, 25))
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=15)
    plotnumber += 1

data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
plt.figure(figsize=(20, 25))
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=15)
    plotnumber += 1

(fig, ax) = plt.subplots(figsize=(15, 10))
sns.boxplot(data=data, width=0.5, ax=ax, fliersize=3)

outlier = data['Pregnancies'].quantile(0.98)
data = data[data['Pregnancies'] < outlier]
outlier = data['BMI'].quantile(0.99)
data = data[data['BMI'] < outlier]
outlier = data['SkinThickness'].quantile(0.99)
data = data[data['SkinThickness'] < outlier]
outlier = data['Insulin'].quantile(0.95)
data = data[data['Insulin'] < outlier]
outlier = data['DiabetesPedigreeFunction'].quantile(0.99)
data = data[data['DiabetesPedigreeFunction'] < outlier]
outlier = data['Age'].quantile(0.99)
data = data[data['Age'] < outlier]
plt.figure(figsize=(20, 25))
plotnumber = 1
for column in data:
    if plotnumber <= 9:
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=15)
    plotnumber += 1

plt.figure(figsize=(16, 8))
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2g', linewidths=1)

X = data.drop(columns=['Outcome'])
y = data['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lr = LogisticRegression()