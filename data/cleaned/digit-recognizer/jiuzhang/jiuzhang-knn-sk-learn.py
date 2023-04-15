import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dir = 'data/input/digit-recognizer/'

def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + 'train.csv')
    print(train.shape)
    X_train = train.values[0:train_row, 1:]
    y_train = train.values[0:train_row, 0]
    Pred_test = pd.read_csv(data_dir + 'test.csv').values
    return (X_train, y_train, Pred_test)
train_row = 5000
(Origin_X_train, Origin_y_train, Origin_X_test) = load_data(data_dir, train_row)
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)
print(Origin_X_train)
import matplotlib
import matplotlib.pyplot as plt
row = 3
print(Origin_y_train[row])
plt.imshow(Origin_X_train[row].reshape((28, 28)))

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
rows = 4
print(classes)
for (y, cls) in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for (i, idx) in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis('off')
        if i == 0:
            plt.title(cls)

from sklearn.model_selection import train_test_split
(X_train, X_vali, y_train, y_vali) = train_test_split(Origin_X_train, Origin_y_train, test_size=0.2, random_state=0)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
ans_k = 0
k_range = range(1, 8)
scores = []
for k in k_range:
    print('k = ' + str(k) + ' begin ')
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)