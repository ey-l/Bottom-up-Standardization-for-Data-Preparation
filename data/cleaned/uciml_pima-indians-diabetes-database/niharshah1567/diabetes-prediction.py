import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
train.head()
train.info()
train.describe()
train_copy = train.copy(deep=True)
train_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = train_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(train_copy.isnull().sum())
train_copy.hist(figsize=(20, 20))
train_copy['Glucose'].fillna(train_copy['Glucose'].mean(), inplace=True)
train_copy['BloodPressure'].fillna(train_copy['BloodPressure'].mean(), inplace=True)
train_copy['SkinThickness'].fillna(train_copy['SkinThickness'].median(), inplace=True)
train_copy['Insulin'].fillna(train_copy['Insulin'].median(), inplace=True)
train_copy['BMI'].fillna(train_copy['BMI'].median(), inplace=True)
train_copy.hist(figsize=(20, 20))
train_copy.shape
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure(figsize=(12, 10))
p = sns.heatmap(train_copy.corr(), annot=True, cmap='RdYlGn')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(train_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = train_copy.Outcome
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42)
logreg = LogisticRegression()