import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print('Library Import Complete')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print('Import Dataset Complete')
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
class_names = {0: 'Not Diabetic', 1: 'Diabetic'}
res_var_analysis = df.Outcome.value_counts().rename(index=class_names)
print(res_var_analysis)
from pandas_profiling import ProfileReport
pp_report = ProfileReport(df)
pp_report.to_file(output_file='pandas_profiling_report.html')
import sweetviz as sv
sw_report = sv.analyze(df)
sw_report.show_html()
from sklearn.model_selection import train_test_split
print('Import Train_Test_Split Complete')
y = df['Outcome']
X = df.loc[:, df.columns != 'Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
print('Test_Train Split Complete')
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
print('Import Accuracy Score and Logistic Regression Libraries Complete')
logreg = LogisticRegression(solver='liblinear')