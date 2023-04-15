import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
numericals = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = ['Outcome']
plt.figure(figsize=(16, 4))
for i in range(0, len(numericals)):
    plt.subplot(1, 8, i + 1)
    sns.boxplot(x=df[numericals[i]])
    plt.tight_layout()
plt.figure(figsize=(16, 10))
for i in range(0, len(numericals)):
    plt.subplot(2, 4, i + 1)
    sns.distplot(df[numericals[i]])
    plt.tight_layout
count_classes = pd.value_counts(df['Outcome'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title('Diabetes Outcome Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
sns.heatmap(df.corr(), annot=True, fmt='.2f')

sns.pairplot(df, diag_kind='kde')
sns.pairplot(df, diag_kind='kde', hue='Outcome')
X = df[numericals]
y = df[target]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state=0, max_iter=400)