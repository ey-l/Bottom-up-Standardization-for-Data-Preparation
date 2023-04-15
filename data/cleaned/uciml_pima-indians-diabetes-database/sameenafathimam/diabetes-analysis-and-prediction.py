import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.shape
df.dtypes
df.isnull().sum().any()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
df.isnull().sum()
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[column].replace(np.nan, df[column].median(), inplace=True)
df.head()
df.Outcome.value_counts()
df.groupby('Outcome')[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']].agg(['max', 'min', 'mean'])
sns.countplot(x='Outcome', data=df)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation heatmap')
df['Age_Group'] = pd.cut(df['Age'], [10, 20, 30, 40, 50, 60], labels=['11-20', '21-30', '31-40', '41-50', '51+'])
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Age_Group', hue='Outcome', ax=ax)
plt.title('Age vs Outcome')
df.drop(['Age_Group'], axis=1, inplace=True)
(fig, ax) = plt.subplots(4, 2, figsize=(16, 16))
sns.kdeplot(data=df['Age'], color='r', shade=True, ax=ax[0][0])
sns.kdeplot(data=df['Pregnancies'], color='b', shade=True, ax=ax[0][1])
sns.kdeplot(data=df['Glucose'], color='r', shade=True, ax=ax[1][0])
sns.kdeplot(data=df['BloodPressure'], color='b', shade=True, ax=ax[1][1])
sns.kdeplot(data=df['SkinThickness'], shade=True, color='r', ax=ax[2][0])
sns.kdeplot(data=df['Insulin'], shade=True, color='b', ax=ax[2][1])
sns.kdeplot(data=df['DiabetesPedigreeFunction'], shade=True, color='r', ax=ax[3][0])
sns.kdeplot(data=df['BMI'], shade=True, color='b', ax=ax[3][1])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeRegressor
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
(train_X, test_X, train_y, test_y) = train_test_split(X, y, test_size=0.3, random_state=1)
classifier = DecisionTreeRegressor(random_state=1)