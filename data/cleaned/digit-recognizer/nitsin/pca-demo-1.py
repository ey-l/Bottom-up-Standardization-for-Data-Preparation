import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df.shape
df.sample()
import matplotlib.pyplot as plt
plt.imshow(df.iloc[13051, 1:].values.reshape(28, 28))
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()