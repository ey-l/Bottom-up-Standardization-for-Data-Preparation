import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
df.describe().T
diabetes = df.copy()
diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diabetes.isnull().sum()
df.hist(figsize=(15, 15))
diabetes['Glucose'].fillna(diabetes['Glucose'].median(), inplace=True)
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(), inplace=True)
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].mean(), inplace=True)
diabetes['Insulin'].fillna(diabetes['Insulin'].median(), inplace=True)
diabetes['BMI'].fillna(diabetes['BMI'].median(), inplace=True)
diabetes.describe().T
diabetes.hist(figsize=(15, 15))
print(diabetes['Outcome'].value_counts())
sns.countplot(diabetes['Outcome'])
plt.figure(figsize=(15, 15))
corr = diabetes.corr()
sns.heatmap(corr, annot=True)
X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
lg = LogisticRegression()