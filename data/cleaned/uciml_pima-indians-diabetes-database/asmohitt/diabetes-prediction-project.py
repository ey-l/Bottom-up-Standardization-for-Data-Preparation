import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
diabetes_dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.describe()
diabetes_dataset.shape
plt.figure(figsize=(8, 5))
sns.countplot(x=diabetes_dataset['Outcome'])
plt.title('Outcome vs count', fontsize=20)
plt.xlabel('Outcome', fontsize=15)
plt.ylabel('Count', fontsize=15)
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
X = diabetes_dataset.iloc[:, :-1].values
y = diabetes_dataset.iloc[:, -1].values
X
y
scalar = StandardScaler()
X = scalar.fit_transform(X)
print(X)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape
y_train.shape
classifier = svm.SVC(C=0.5, kernel='linear')