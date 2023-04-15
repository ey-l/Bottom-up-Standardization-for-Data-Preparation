import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isna().sum()
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
robscaler = RobustScaler()
X = robscaler.fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=6)
logreg = LogisticRegression()