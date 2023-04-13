import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe().T
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Pandas profiling report ', html={'style': {'full_width': True}})
profile.to_notebook_iframe()
df.isna().sum().plot(kind='bar')
pass
pass
ax.set_facecolor('#fafafa')
ax.set(xlim=(-0.05, 200))
pass
pass
pass
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.isna().sum().plot(kind='bar')
Diabetes = df[df['Outcome'] != 0]
No_diab = df[df['Outcome'] == 0]
Diabetes['Glucose'].mean()
Diabetes.hist(figsize=(15, 12))
No_diab.hist(figsize=(15, 12))
df.loc[(df['Outcome'] == 0) & df['Insulin'].isnull(), 'Insulin'] = No_diab['Insulin'].mean()
df.loc[(df['Outcome'] == 1) & df['Insulin'].isnull(), 'Insulin'] = Diabetes['Insulin'].mean()
df.loc[(df['Outcome'] == 0) & df['Glucose'].isnull(), 'Glucose'] = No_diab['Glucose'].mean()
df.loc[(df['Outcome'] == 1) & df['Glucose'].isnull(), 'Glucose'] = Diabetes['Glucose'].mean()
df.loc[(df['Outcome'] == 0) & df['SkinThickness'].isnull(), 'SkinThickness'] = No_diab['SkinThickness'].mean()
df.loc[(df['Outcome'] == 1) & df['SkinThickness'].isnull(), 'SkinThickness'] = Diabetes['SkinThickness'].mean()
df.loc[(df['Outcome'] == 0) & df['BloodPressure'].isnull(), 'BloodPressure'] = No_diab['BloodPressure'].mean()
df.loc[(df['Outcome'] == 1) & df['BloodPressure'].isnull(), 'BloodPressure'] = Diabetes['BloodPressure'].mean()
df.loc[(df['Outcome'] == 0) & df['BMI'].isnull(), 'BMI'] = No_diab['BMI'].mean()
df.loc[(df['Outcome'] == 1) & df['BMI'].isnull(), 'BMI'] = Diabetes['BMI'].mean()
df.isnull().sum().plot(kind='bar')
df.hist(figsize=(15, 12))
df.columns
print(df.shape)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train_std = ss.fit_transform(x_train)
x_test_std = ss.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [5, 10, 15, 25, 30, 50]}
grid_knn = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=10, refit=True, n_jobs=-1)