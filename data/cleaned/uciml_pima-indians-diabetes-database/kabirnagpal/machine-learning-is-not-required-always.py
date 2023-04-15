import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.loc[df.Pregnancies > 6, 'Pregnancies'] = 6
df.loc[df.Glucose < 70, 'Glucose'] = 70
df.loc[df.BloodPressure < 60, 'BloodPressure'] = 60
df.loc[df.BMI < 18, 'BMI'] = 18
df.loc[df.BMI > 40, 'BMI'] = 40
df.describe()
plt.title('Ratio of Healthy and Diabetic Patients in Dataset')
plt.pie(df['Outcome'].value_counts(), autopct='%.2f')
plt.legend(['Healthy', 'Diabetic'])


def plot(s):
    sns.kdeplot(df.loc[df.Outcome == 0, s])
    sns.kdeplot(df.loc[df.Outcome == 1, s])
    plt.legend(['Healthy', 'Diabetic'])
    plt.ylabel('Number of Patients')
    plt.yticks([])

for i in df.columns[:-1]:
    plot(i)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)

from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.metrics import *
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.2)
clf = DecisionTreeClassifier()