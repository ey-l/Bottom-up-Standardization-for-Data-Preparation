import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
df.info()
sns.pairplot(df)
plt.pyplot.figure(figsize=(16, 16))
sns.heatmap(df.corr(), annot=True)
sns.boxplot(x='Outcome', y='Glucose', data=df)
X = df.drop(['Outcome', 'Age'], axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
model = GaussianNB()