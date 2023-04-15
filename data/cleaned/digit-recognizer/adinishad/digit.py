import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
train.head()
test.head()
train.shape
test.shape
from sklearn.model_selection import train_test_split
X = train.iloc[:, 1:]
y = train.iloc[:, 0]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape
import matplotlib.pyplot as plt
img = X_train.iloc[1]
img = np.asarray(img)
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train.iloc[3, 0])
plt.axis('off')

from sklearn import svm
model = svm.SVC()