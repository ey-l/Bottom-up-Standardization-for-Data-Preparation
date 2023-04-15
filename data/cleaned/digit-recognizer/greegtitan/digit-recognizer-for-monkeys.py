import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
BASE_PATH = 'data/input/digit-recognizer/'
TEST_FILE = os.path.join(BASE_PATH, 'test.csv')
TRAIN_FILE = os.path.join(BASE_PATH, 'train.csv')
test_dataset = pd.read_csv(TEST_FILE)
test_dataset.head()
train_dataset = pd.read_csv(TRAIN_FILE)
train_dataset.head()
X = train_dataset.drop('label', axis=1)
y = train_dataset['label']
print('X shape:', X.shape)
print('y shape:', y.shape)

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(20, 2))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    number_index = y.loc[y == i].index[0]
    ax.imshow(X.loc[number_index].values.reshape(28, 28), cmap='binary')
    ax.axis('off')
    ax.set_title(i, color='red')
fig = plt.figure(figsize=(8, 8))
random_int = np.random.randint(0, X.shape[0], size=9, dtype=int)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(X.loc[random_int[i]].values.reshape(28, 28), cmap='binary')
    ax.axis('off')
print('X train missing data:', X.isna().sum().sum())
print('y train missing data:', y.isna().sum().sum())
print('test dataset missing data:', test_dataset.isna().sum().sum())
print('Singular data:')
print(X.iloc[0].values)
random_int = 69
(fig, ax) = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow((X.iloc[69].values / 255).reshape(28, 28), cmap='binary')
ax[0].axis('off')
ax[1].imshow(X.iloc[69].values.reshape(28, 28), cmap='binary')
ax[1].axis('off')

X = X / 255
X.iloc[0][X.iloc[0].values != 0].values[:10]
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.2)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
knc = KNeighborsClassifier()