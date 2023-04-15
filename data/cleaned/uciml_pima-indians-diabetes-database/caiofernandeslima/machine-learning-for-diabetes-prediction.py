import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe()

def counting_zeros(columnName):
    print('The number of zeros in %s is %d' % (columnName, len(df[df[columnName] == 0])))
for column in df.columns:
    counting_zeros(column)
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns:
    df[column] = df[column].replace(to_replace=0, value=np.nan)
df.describe()
plt.figure(figsize=(12, 10))
sns.heatmap(data=df.corr(), annot=True)


def boxes_plot(data):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    plt.figure(figsize=(15, 15))
    for i in range(len(columns)):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(x=columns[i], data=data)

boxes_plot(df)

def histograms_plot(data):
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    plt.figure(figsize=(20, 20))
    for i in range(len(columns)):
        plt.subplot(2, 4, i + 1)
        sns.histplot(data=data, x=columns[i], kde=True, stat='count')

histograms_plot(df)
y = df['Outcome']
X = df.drop(labels=['Outcome'], axis=1)
X.head()
y.head()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
X_train_copy = X_train.copy()

def print_mean_median_mode(column, df):
    print('mean', df[column].mean())
    print('median', df[column].median())
    print('mode', df[column].mode())
print_mean_median_mode('Glucose', X_train_copy)
X_train_copy['Glucose'] = X_train_copy['Glucose'].fillna(value=X_train_copy['Glucose'].mean().astype('int64'))
print_mean_median_mode('BloodPressure', X_train_copy)
X_train_copy['BloodPressure'] = X_train_copy['BloodPressure'].fillna(value=X_train_copy['BloodPressure'].median())
print_mean_median_mode('SkinThickness', X_train_copy)
X_train_copy['SkinThickness'] = X_train_copy['SkinThickness'].fillna(value=X_train_copy['SkinThickness'].median())
print_mean_median_mode('Insulin', X_train_copy)
X_train_copy['Insulin'] = X_train_copy['Insulin'].fillna(value=X_train_copy['Insulin'].median())
print_mean_median_mode('BMI', X_train_copy)
X_train_copy['BMI'] = X_train_copy['BMI'].fillna(value=X_train_copy['BMI'].median())
X_train_copy.isnull().sum()
for column in X_train_copy.columns:
    X_train_copy[column] = X_train_copy[column] - np.min(X_train_copy[column].values) + 1
for column in X_train_copy.columns:
    X_train_copy[column] = np.log(X_train_copy[column])
X_train_copy.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()