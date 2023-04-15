import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.info())
data.head()
import seaborn as sns
data_corr = data.corr()
plt.figure(figsize=(6, 6))
sns.heatmap(data_corr, annot=True)

len(data[data.Outcome == 1])
sns.barplot(x=['Negative', 'Positive'], y=data.Outcome.value_counts(), palette='rocket')
plt.figure(figsize=(7, 4))
ax = sns.boxplot(x=data.Outcome, y=data.Pregnancies, data=data, palette='rocket')

plt.figure(figsize=(7, 4))
ax = sns.boxplot(x=data.Outcome, y=data.Age, data=data, palette='rocket')

plt.figure(figsize=(7, 4))
ax = sns.boxplot(x=data.Outcome, y=data.BMI, data=data, palette='rocket')

plt.figure(figsize=(7, 4))
ax = sns.boxplot(x=data.Outcome, y=data.Glucose, data=data, palette='rocket')

x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train_scl = ss.fit_transform(x_train)
x_test_scl = ss.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()