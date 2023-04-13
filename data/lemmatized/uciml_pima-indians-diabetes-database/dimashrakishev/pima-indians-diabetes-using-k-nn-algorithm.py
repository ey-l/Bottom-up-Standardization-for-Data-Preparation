import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(5)
print(df.shape)
X = df.drop(columns='Outcome')
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_accuracies = []
train_accuracies = []
for n_neighbors in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors)