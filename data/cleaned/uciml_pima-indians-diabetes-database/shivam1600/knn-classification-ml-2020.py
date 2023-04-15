import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X[:5]
y[:5]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for (i, k) in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)