import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isna().sum()
df.dtypes
df.describe()
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='Pastel1')
plt.title('Correlation of Diabetes', fontsize=20)
df.columns
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='Age', y='Glucose', color='Black')
plt.figure(figsize=(20, 10))
sns.boxplot(data=df, x='Age', y='BloodPressure')
plt.figure(figsize=(20, 10))
sns.barplot(data=df, x='Age', y='Pregnancies')
plt.figure(figsize=(20, 20))
sns.boxplot(data=df, x='Glucose', y='Insulin')
plt.figure(figsize=(10, 10))
sns.displot(df['Outcome'])
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
(len(X_train), len(X_test), len(y_train), len(y_test))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
lr = LogisticRegression(random_state=42)