import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diab_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab_df.head()
diab_df.info()
diab_df.describe()
plt.figure(figsize=(16, 9))
sns.heatmap(diab_df)
diab_df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(diab_df.corr(), annot=True, cmap='coolwarm', linewidths=2)
diab_df.shape
diab_df.isnull().sum()
X = diab_df.drop(['Outcome'], axis=1)
X.head()
y = diab_df['Outcome']
y.head()
y.value_counts()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(penalty='l1', solver='liblinear')