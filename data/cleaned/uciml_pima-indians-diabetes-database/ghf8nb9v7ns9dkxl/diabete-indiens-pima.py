import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
discrete_features = ['Outcome']
continuous_features = list(set(df.columns) - set(discrete_features))
df.head()
df.info()
df.count()
df.min()
plt.figure(figsize=(10, 10))
sns.heatmap(abs(df.corr()))
plt.figure(figsize=(20, 5))
sns.swarmplot(x='Outcome', y='Pregnancies', data=df, hue='Outcome')
plt.figure(figsize=(30, 5))
df2 = df.drop(['Glucose', 'Pregnancies', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'BloodPressure'], axis=1)
df
sns.countplot(x='Age', data=df2, hue='Outcome')
plt.figure(figsize=(15, 5))
sns.boxplot(data=df[continuous_features])
sns.displot(x='Glucose', data=df, hue='Outcome', kde=True)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()